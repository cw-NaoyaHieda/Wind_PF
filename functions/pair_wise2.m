function [pw_weight] = pair_wise2(phi1, sig_rho, mu_rho,filter_X, rho1, filter_weight, sm_weight)
  [dT N] = size(filter_weight);
  if N > 1000 | dT > 100000
    pw_weight = zeros(N, N, dT);
    for dt = 2:(dT - 1)
      bunbo = filter_weight(dt - 1,:) * (normpdf([filter_X(dt,:)], [phi1 *  filter_X(dt - 1, :)]', sqrt(1 - phi1)) .*  normpdf(rho1(dt+1,:), [0.95 .* (tanh(sig_rho .* filter_X(dt,:) + mu_rho)+1)/2]', 0.95*sqrt(1 - phi1^2)));
      pw_weight(:,:,dt) = gather(normpdf([filter_X(dt,:)], [phi1 *  filter_X(dt - 1, :)]', sqrt(1 - phi1)) .* normpdf(rho1(dt+1,:), [0.95 .* (tanh(sig_rho .* filter_X(dt,:) + mu_rho)+1)/2]', 0.95*sqrt(1 - phi1^2)) .* (filter_weight(dt - 1,:)' * sm_weight (dt,:) ./ bunbo));
    end
  else
    pw_weight = ones(N, N, dT,'gpuArray');
    for dt = 2:(dT - 1)
      bunbo = filter_weight(dt - 1,:) * (normpdf([filter_X(dt,:)], [phi1 *  filter_X(dt - 1, :)]', sqrt(1 - phi1)).*  normpdf(rho1(dt+1,:), [0.95 .* (tanh(sig_rho .* filter_X(dt,:) + mu_rho)+1)/2]', 0.95*sqrt(1 - phi1^2)));
      pw_weight(:,:,dt) = normpdf([filter_X(dt,:)], [phi1 *  filter_X(dt - 1, :)]', sqrt(1 - phi1)) .* normpdf(rho1(dt+1,:), [0.95 .* (tanh(sig_rho .* filter_X(dt,:) + mu_rho)+1)/2]', 0.95*sqrt(1 - phi1^2)) .* (filter_weight(dt - 1,:)' * sm_weight (dt,:) ./ bunbo);
    end
  end
  
end