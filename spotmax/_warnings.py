def warn_background_value_is_zero(logger_func, logger_warning_report=None):
    text = (
        'Background value is 0 --> '
        'spot center intensity to background ratio is infinite'
    )
    print('')
    logger_func(f'[WARNING]: {text}')
    
    if logger_warning_report is None:
        return
    
    logger_warning_report(text)