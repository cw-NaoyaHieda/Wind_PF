function [pw_weight] = pair_wise(phi1, filter_X, filter_weight, sm_weight)
  [dT N] = size(filter_weight);
  if N > 1000 | dT > 100000
    pw_weight = zeros(N, N, dT);
    for dt = 2:(dT - 1)
      bunbo = filter_weight(dt - 1,:) * normpdf([filter_X(dt,:)], [phi1 *  filter_X(dt - 1, :)]', sqrt(1 - phi1));
      pw_weight(:,:,dt) = gather(normpdf([filter_X(dt,:)], [phi1 *  filter_X(dt - 1, :)]', sqrt(1 - phi1)) .* (filter_weight(dt - 1,:)' .* sm_weight (dt,:) ./ bunbo));
    end
  else
    pw_weight = ones(N, N, dT,'gpuArray');
    filter_X = gpuArray(filter_X);
    for dt = 2:(dT - 1)
      bunbo = filter_weight(dt - 1,:) * normpdf([filter_X(dt,:)], [phi1 *  filter_X(dt - 1, :)]', sqrt(1 - phi1));
      pw_weight(:,:,dt) = normpdf([filter_X(dt,:)], [phi1 *  filter_X(dt - 1, :)]', sqrt(1 - phi1)) .* (filter_weight(dt - 1,:)' .* sm_weight (dt,:) ./ bunbo);
    end
  end

end
