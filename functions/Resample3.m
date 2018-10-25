function [newdata1] = Resample3(data1, weight, NofSample)
  cs_weight = cumsum(weight);
  ch = rand(1,NofSample) < cs_weight';
  
  y =  NofSample - sum(ch) + 1;
  
  newdata1 = data1(y);
 
end
