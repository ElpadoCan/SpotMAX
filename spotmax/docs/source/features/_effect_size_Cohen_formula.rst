.. role:: m(math)

  .. math::
    
    \frac{\mathrm{mean}(P) - \mathrm{mean}(N)}{\mathrm{std}(NP)}

  where :m:`\mathrm{std}(NP)` is the pooled standard deviation of the spots 
  and background intensities and it is calculated as follows:

  .. math:: 

    \mathrm{std}(NP) = \sqrt{\frac{(n_P - 1)s_P^2 + (n_N - 1)s_N^2}{n_P + n_N - 2}}

  where :m:`n_P` and :m:`n_N` are the spot and background sample sizes, while 
  :m:`s_P` and :m:`s_N` are the spot and background standard deviations, 
  respectively. 