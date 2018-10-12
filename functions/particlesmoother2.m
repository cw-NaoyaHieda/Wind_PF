function [smwt] = particlesmoother2(phi1, sig_rho , mu_rho, pfOut1, rho1, wt)
  [Xsize Ysize] = size(pfOut1);
  
  pfOut1 = gpuArray(pfOut1);
  wt = gpuArray(wt);
  smwt = gpuArray(wt);
  rho1 = gpuArray(rho1);
  
  
  for dt = (Xsize - 1):-1:1
    sm_table1 = normpdf(pfOut1(dt+1,:), [phi1 *  pfOut1(dt, :)]', sqrt(1 - phi1^2));
    sm_table2 = normpdf(rho1(dt+1,:),  [0.95*(tanh(sig_rho * phi1 * pfOut1(dt,:) + mu_rho)+1)/2]', 0.95*sqrt(1 - phi1^2));
    sm_table = sm_table1 .* sm_table2;
    bunsi = smwt(dt+1, :) .* sm_table;
    bunbo = wt(dt, :) * sm_table;
    smwt(dt,:) = wt(dt,:)' .* (bunsi * (1./bunbo)');
  end
end