addpath('functions')
rng(1000)
clear global;
reset(gpuDevice(1));
%時系列数
N = 300;

%塩浜先生がお作りしたものを使用しています
sample = csvread("sample_1000.csv",1,1);
sample = sample(1:300,:);
%観測変数の取り出し(観測変数としては使わないものもある)
theta = sample(:,1);
v =sample(:,2);
rho =sample(:,3);
alpha =sample(:,4);

y = theta;
v = v;
r = rho;
alp = alpha;


%パラメータ設定 微妙にずらします
phi1 = 0.97;
gam = 3; % constants in wind speed
mu_g = 0.0; % location in wind direction for transition
mu_f = 0.0; % location in wind direction for marginal
rho_f = 0.1; % consentration in wind direction for marginal
V = 20;
mu_rho = 0.5;%given
sig_rho=1;%given
par1 = [phi1 gam mu_g mu_f rho_f V];

opt_params = zeros(6,8);
%初期パラメータの設定
opt_params(1,:) = [0,0,par1];

[pfOut1, pfOut2, wt, pfOut1_mean, pfOut2_mean,rho1] = particlefilter_reallycorrect_notexpress(par1,mu_rho,sig_rho,y, v, r, alp, 2000);
phi1 = par1(1);
i
    [smwt] = particlesmoother(phi1, pfOut1, wt);
    %[smwt] = particlesmoother2(phi1, sig_rho, mu_rho, pfOut1, rho1, wt);
    i
    sm_mean = diag(smwt(2:(N+1),:) * gpuArray(pfOut1(2:(N+1),:)'));
    i+N
    pw_weight = pair_wise(phi1, pfOut1, wt, smwt);
    %pw_weight = pair_wise2(phi1, sig_rho, mu_rho, pfOut1, rho1, wt, smwt);
    i-N
    
    
    
    PMCEM_calc = @(par1)Q_calc_now(par1, pfOut1, pfOut2, rho1,pw_weight, smwt, y, v);
   