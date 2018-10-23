function [Q] = Q_calc_now(par, pfOut1, pfOut2, rho1, pw_weight, sm_weight, y, v)
  phi1 = par(1); % AR in state of wind speed 0.99
  gam  = par(2); % constants in log wind speed
  mu_g = par(3); % location in wind direction for transition
  mu_f = par(4); % location in wind direction for marginal
  rho_f = par(5); % consentration in wind direction for marginal
  V = par(6);

  Q_state = 0;
  Q_obeserve = 0;
  first_state = 0;

  [dT  nParticle]= size(pfOut1);

  pfOut1 = gpuArray(pfOut1);



  for dt = 2:(dT - 1)
    Q_state = Q_state + sum(sum(gpuArray(pw_weight(:,:,dt)) .* log(normpdf([pfOut1(dt,:)], [phi1 *  pfOut1(dt - 1, :)]', sqrt(1 - phi1^2)))));
    tmp1 = arrayfun(@(theta, rho_g) d_conditional_WJ(y(dt-1), theta, mu_g, rho_g, mu_f, rho_f, 1), pfOut2(dt-1,:), rho1(dt-1,:));
    tmp2 = gampdf(v(dt-1)./(gam*exp( pfOut1(dt,:) /2)) , V, 1/V) ./ (gam*exp(pfOut1(dt,:)/2));
    Q_obeserve = Q_obeserve + sm_weight(dt,:) * log(tmp1)' + sm_weight(dt,:) * log(tmp2)' ;


  end
  %first_state = sm_weight(1,:) * log(normpdf([pfOut1(1,:)], phi1 *  X_0_est, sqrt(1 - phi^2)))';
  Q = gather(-Q_state - Q_obeserve);
end
