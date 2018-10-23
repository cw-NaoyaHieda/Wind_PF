check_para = 1;

if check_para == 1
    phi_check = zeros(10,2);
    phis = [0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99];
    for i = 1:10
       par1(1) = phis(i);
       phi_check(i,:) = [phis(i);,PMCEM_calc(par1)]
    end
    par1(1) = 0.97;
end

%‚±‚ê‚Í‚æ‚³‚°
if check_para == 2
    gam_check = zeros(9,2);
    gams = [1,1.5,2,2.5,3,3.5,4,4.5,5];
    for i = 1:9
       par1(2) = gams(i);
       gam_check(i,:) = [gams(i);,PMCEM_calc(par1)]
    end
    par1(2) = 3;
end

if check_para == 3
    mu_g_check = zeros(10,2);
    mu_gs = [-2,-1.5,-1,-0.5,0,0.5,1,1.5,2];
    for i = 1:9
       par1(3) = mu_gs(i);
       mu_g_check(i,:) = [mu_gs(i);,PMCEM_calc(par1)]
    end
    par1(3) = 0;
end

if check_para == 4
    mu_f_check = zeros(10,2);
    mu_fs = [-2,-1.5,-1,-0.5,0,0.5,1,1.5,2];
    for i = 1:9
       par1(4) = mu_fs(i);
       mu_f_check(i,:) = [mu_fs(i);,PMCEM_calc(par1)]
    end
    par1(4) = 0;
end

if check_para == 4
    mu_f_check = zeros(10,2);
    mu_fs = [-2,-1.5,-1.-0.5,0,0.5,1,1.5,2];
    for i = 1:9
       par1(4) = mu_fs(i);
       mu_f_check(i,:) = [mu_fs(i);,PMCEM_calc(par1)]
    end
    par1(4) = 0;
end