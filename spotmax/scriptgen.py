"""Generate Python entry points for running INI-configured analyses."""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid

import numpy as np
import pandas as pd

from . import config, core, io, transformations


def _loaded(params, section, anchor):
    option = params[section][anchor]
    value = option.get("loadedVal")
    if value is None:
        value = option.get("initialVal")
    return value


def _model_for_kernel(kernel, model_class):
    params = kernel._params
    prediction_method = _loaded(
        params, "Spots channel", "spotPredictionMethod"
    )
    method_keys = {
        "Thresholding": "custom",
        "spotMAX AI": "neural_network",
        "BioImage.IO model": "bioimageio_model",
        "Spotiflow": "spotiflow",
    }
    try:
        segmentation_method = method_keys[prediction_method]
    except KeyError as error:
        raise ValueError(
            f"Unsupported spots segmentation method: {prediction_method!r}"
        ) from error

    segmentation_kwargs = {
        "do_remove_hot_pixels": _loaded(
            params, "Pre-processing", "removeHotPixels"
        ),
        "gauss_sigma": _loaded(params, "Pre-processing", "gaussSigma"),
        "do_sharpen": _loaded(params, "Pre-processing", "sharpenSpots"),
        "do_aggregate": _loaded(params, "Pre-processing", "aggregate"),
        "thresh_only_inside_objs_intens": _loaded(
            params, "Pre-processing", "thresholdWithObjsMask"
        ),
        "min_spot_mask_size": _loaded(
            params, "Spots channel", "minSizeSpotMask"
        ),
        "thresholding_method": _loaded(
            params, "Spots channel", "spotThresholdFunc"
        ),
        "use_gpu": kernel._get_use_gpu(),
        "nnet_model": kernel.nnet_model,
        "nnet_params": kernel.nnet_params,
        "bioimageio_model": kernel.bioimageio_model,
        "bioimageio_params": kernel.bioimageio_params,
        "spotiflow_model": kernel.spotiflow_model,
        "spotiflow_params": kernel.spotiflow_params,
    }
    zyx_radii = kernel.metadata["zyxResolutionLimitPxl"]
    spot_footprint = transformations.get_local_spheroid_mask(
        [radius / 2 for radius in zyx_radii]
    )
    detection_kwargs = {
        "detection_method": _loaded(
            params, "Spots channel", "spotDetectionMethod"
        ),
        "spot_footprint": spot_footprint,
    }
    return model_class(
        spots_zyx_radii_pxl=zyx_radii,
        zyx_voxel_size=kernel.metadata["zyxVoxelSize"],
        zyx_spot_min_vol_um=kernel.metadata["zyxResolutionLimitUm"],
        segmentation_kwargs=segmentation_kwargs,
        segmentation_method=segmentation_method,
        detection_kwargs=detection_kwargs,
    )


def _write_generated_ini(params_ini_path, coords_endname):
    fd, work_ini_path = tempfile.mkstemp(
        prefix=f"{os.path.splitext(os.path.basename(params_ini_path))[0]}_model_",
        suffix=".ini",
        dir=os.path.dirname(params_ini_path) or os.curdir,
    )
    os.close(fd)
    shutil.copy2(params_ini_path, work_ini_path)

    config_parser = config.ConfigParser()
    config_parser.read(work_ini_path, encoding="utf-8")
    section = "File paths and channels"
    option = config._filepaths_params()["inputDfSpotsEndname"]["desc"]
    if not config_parser.has_section(section):
        config_parser.add_section(section)
    config_parser[section][option] = coords_endname
    with open(work_ini_path, "w", encoding="utf-8") as file:
        config_parser.write(file)
    return work_ini_path


def _write_prediction_coordinates(kernel, model, exp_path, exp_info,
                                  pos_foldername, transformed_nnet_data,
                                  coords_endname):
    images_path = os.path.join(exp_path, pos_foldername, "Images")
    data = kernel.get_data_from_images_path(
        images_path,
        exp_info["spotsEndName"],
        exp_info["refChEndName"],
        exp_info["segmEndName"],
        exp_info["spotChSegmEndName"],
        exp_info["refChSegmEndName"],
        exp_info["lineageTableEndName"],
        "",
    )
    if data is None or "spots_ch" not in data:
        raise FileNotFoundError(
            f"Could not load the spots channel in {images_path!r}."
        )

    data = kernel._extend_3D_segm_in_z(
        data, _loaded(kernel._params, "Pre-processing", "extend3DsegmRange")
    )
    spots_data = data["spots_ch"]
    segm_data = data["segm"]
    spots_segm_data = data.get("spots_ch_segm")
    lineage_table = data.get("lineage_table")
    transformed_time_data = kernel.check_preprocess_data_nnet_across_time(
        spots_data, transformed_spots_ch_nnet=transformed_nnet_data
    )
    stop_frame = min(kernel.metadata["stopFrameNum"], len(spots_data))
    crop_offset = np.asarray(data.get("crop_to_global_coords", (0, 0, 0)))

    frame_tables = []
    for frame_i in range(stop_frame):
        lab = segm_data[frame_i]
        if lab is None or not np.any(lab):
            # A frame entry keeps Kernel from treating this frame as absent.
            frame_tables.append(pd.DataFrame({
                "frame_i": [frame_i], "z": [-1], "y": [-1], "x": [-1]
            }))
            continue

        model.frame_i = frame_i
        if lineage_table is not None:
            try:
                frame_lineage = lineage_table.loc[[frame_i]].droplevel(0)
            except KeyError:
                frame_lineage = None
        else:
            frame_lineage = None
        model.segmentation_kwargs["lineage_table"] = frame_lineage

        if transformed_time_data is None:
            model.segmentation_kwargs["nnet_input_data"] = None
        else:
            model.segmentation_kwargs["nnet_input_data"] = transformed_time_data[
                frame_i
            ]

        if spots_segm_data is None:
            segmentation_mask = None
        else:
            segmentation_mask = spots_segm_data[frame_i]

        df_coords = model.predict(
            spots_data[frame_i],
            lab=lab,
            segmentation_mask=segmentation_mask,
        )
        if df_coords.empty:
            frame_tables.append(pd.DataFrame({
                "frame_i": [frame_i], "z": [-1], "y": [-1], "x": [-1]
            }))
            continue

        frame_coords = df_coords.loc[:, ["z", "y", "x"]].copy()
        frame_coords[["z", "y", "x"]] = (
            frame_coords[["z", "y", "x"]].to_numpy() + crop_offset
        )
        frame_coords.insert(0, "frame_i", frame_i)
        frame_tables.append(frame_coords.reset_index(drop=True))

    if frame_tables:
        df_coords = pd.concat(frame_tables, ignore_index=True)
    else:
        df_coords = pd.DataFrame(columns=["frame_i", "z", "y", "x"])
    spotmax_out_path = os.path.join(
        os.path.dirname(images_path), "spotMAX_output"
    )
    os.makedirs(spotmax_out_path, exist_ok=True)
    df_coords.to_csv(
        os.path.join(spotmax_out_path, coords_endname), index=False
    )


def run_ini_batch_with_model(params_ini_path, model_class):
    """Run an INI batch analysis using ``model_class.predict`` per frame.

    Spot coordinates are first generated with the high-level image model. The
    normal Kernel run then consumes those coordinates and performs the rest of
    the configured analysis, including feature calculation, SpotFIT, reference
    channel processing, and the standard SpotMAX output saving.

    The generated working INI and per-position coordinate CSVs are retained so
    the run can be reproduced from its output folder.
    """
    params_ini_path = os.path.abspath(os.fspath(params_ini_path))
    if not params_ini_path.lower().endswith(".ini"):
        raise ValueError("params_ini_path must point to an .ini file.")
    if not os.path.isfile(params_ini_path):
        raise FileNotFoundError(params_ini_path)

    coords_endname = f"spotmax_model_coords_{uuid.uuid4().hex}.csv"
    work_ini_path = _write_generated_ini(params_ini_path, "")

    raw_params = config.analysisInputsParams(work_ini_path)
    force_defaults = _loaded(
        raw_params, "Configuration", "forceDefaultValues"
    )
    if force_defaults is None:
        force_defaults = False

    kernel = core.Kernel(is_cli=False)
    kernel._force_default = bool(force_defaults)
    proceed, missing_params = kernel.init_params(work_ini_path)
    if not proceed:
        raise RuntimeError(
            "SpotMAX could not initialize the generated working INI. "
            f"Missing parameters: {missing_params}"
        )

    model = _model_for_kernel(kernel, model_class)
    for exp_paths in kernel.exp_paths_list:
        for exp_path, exp_info in exp_paths.items():
            pos_foldernames = exp_info["pos_foldernames"]
            transformed_by_pos = kernel.check_preprocess_data_nnet_across_exp(
                exp_path,
                pos_foldernames,
                exp_info["spotsEndName"],
            )
            if transformed_by_pos is None:
                transformed_by_pos = {}
            for pos_foldername in pos_foldernames:
                _write_prediction_coordinates(
                    kernel,
                    model,
                    exp_path,
                    exp_info,
                    pos_foldername,
                    transformed_by_pos.get(pos_foldername),
                    coords_endname,
                )

    # Point the second Kernel pass to the per-frame coordinates produced by
    # pipe.Model.predict.
    config_parser = config.ConfigParser()
    config_parser.read(work_ini_path, encoding="utf-8")
    section = "File paths and channels"
    option = config._filepaths_params()["inputDfSpotsEndname"]["desc"]
    config_parser[section][option] = coords_endname
    with open(work_ini_path, "w", encoding="utf-8") as file:
        config_parser.write(file)

    disable_report = bool(_loaded(kernel._params, "Configuration", "disableFinalReport"))
    report_folder = _loaded(kernel._params, "Configuration", "pathToReport") or ""
    report_filename = _loaded(kernel._params, "Configuration", "reportFilename") or ""
    report_filepath = ""
    if not disable_report and report_filename == "do_not_save":
        disable_report = True
    if not disable_report:
        report_filepath = kernel._check_report_filepath(
            report_folder,
            work_ini_path,
            report_filename=report_filename,
            force_default=bool(force_defaults),
        )
        if report_filepath is None:
            raise RuntimeError("No final report path was selected.")
        if report_filepath == "do_not_save":
            disable_report = True
        else:
            if not os.path.isabs(report_filepath):
                report_filepath = io.get_abspath(
                    report_filepath,
                    src_folderpath=os.path.dirname(work_ini_path),
                )
            report_filepath = kernel._check_exists_validate_report_file(
                report_filepath,
                work_ini_path,
                force_default=bool(force_defaults),
            )
            if report_filepath is None:
                raise RuntimeError("No final report path was selected.")
            if report_filepath == "do_not_save":
                disable_report = True

    kernel.run(
        work_ini_path,
        num_numba_threads=int(_loaded(
            kernel._params, "Configuration", "numbaNumThreads"
        )),
        force_default_values=bool(force_defaults),
        force_close_on_critical=bool(_loaded(
            kernel._params, "Configuration", "raiseOnCritical"
        )),
        disable_final_report=disable_report,
        report_filepath=report_filepath,
    )


def create_python_script_from_ini(params_ini_path, script_path=None):
    """Create a Python script that runs an INI batch through ``pipe.Model``.

    The generated script calls ``pipe.Model.predict`` once for each frame to
    generate spot coordinates, then lets ``core.Kernel`` perform the remaining
    INI-configured analysis and save the usual SpotMAX outputs.

    Parameters
    ----------
    params_ini_path : path-like
        SpotMAX analysis parameter INI file.
    script_path : path-like, optional
        Destination path. By default, writes ``<ini-stem>_python.py`` next to
        the INI file.

    Returns
    -------
    str
        Absolute path of the generated script.
    """
    params_ini_path = os.path.abspath(os.fspath(params_ini_path))
    if not params_ini_path.lower().endswith(".ini"):
        raise ValueError("params_ini_path must point to an .ini file.")
    if not os.path.isfile(params_ini_path):
        raise FileNotFoundError(params_ini_path)

    if script_path is None:
        script_path = os.path.splitext(params_ini_path)[0] + "_python.py"
    script_path = os.path.abspath(os.fspath(script_path))
    if os.path.normcase(script_path) == os.path.normcase(params_ini_path):
        raise ValueError("script_path must not overwrite the source INI file.")
    if os.path.exists(script_path):
        raise FileExistsError(
            f"Refusing to overwrite the existing Python script: {script_path}"
        )
    os.makedirs(os.path.dirname(script_path), exist_ok=True)

    script_text = (
        "\"\"\"Generated by spotmax.config.create_python_script_from_ini.\"\"\"\n"
        "from spotmax import pipe\n"
        "from spotmax.scriptgen import run_ini_batch_with_model\n\n"
        f"PARAMS_INI_PATH = {params_ini_path!r}\n\n"
        "# The runner calls pipe.Model.predict(image, lab=...) for every frame,\n"
        "# then runs the remaining configured analysis with core.Kernel.\n"
        "if __name__ == '__main__':\n"
        "    run_ini_batch_with_model(\n"
        "        PARAMS_INI_PATH, model_class=pipe.Model\n"
        "    )\n"
    )
    with open(script_path, "w", encoding="utf-8") as file:
        file.write(script_text)
    return script_path
