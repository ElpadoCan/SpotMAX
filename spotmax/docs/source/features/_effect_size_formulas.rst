.. role:: m(math)

* **Glass**: column name ``spot_vs_backgr_effect_size_glass``. 
  Formula: 

  .. math::
    
    \frac{\mathrm{mean}(P) - \mathrm{mean}(N)}{\mathrm{std}(N)}

* **Cohen**: column name ``spot_vs_backgr_effect_size_cohen``. 
  Formula:
  
  .. math::
    
    \frac{\mathrm{mean}(P) - \mathrm{mean}(N)}{\mathrm{std}(NP)}

  where :m:`\mathrm{std}(NP)` is the pooled standard deviation of the spots 
  and background intensities and it is calculated as follows:

  .. math:: 

    \mathrm{std}(NP) = \sqrt{\frac{(n_P - 1)s_P^2 + (n_N - 1)s_N^2}{n_P + n_N - 2}}

  where :m:`n_P` and :m:`n_N` are the spot and background sample sizes, while 
  :m:`s_P` and :m:`s_N` are the spot and background standard deviations, 
  respectively. 

* **Hedge**: column name ``spot_vs_backgr_effect_size_hedge``. 
  Formula:  
  
  .. math::
    
    d \cdot c_f
  
  where :m:`d` is the Cohen's effect size and 
  :m:`c_f = 1 - 3/(4\Delta n - 9)` with :m:`\Delta n` being the 
  difference between the spot and background sample sizes. 