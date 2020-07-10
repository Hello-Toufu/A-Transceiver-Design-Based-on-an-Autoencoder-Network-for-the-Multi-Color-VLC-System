%===============================================================================
% # Author            :   ZDF
% # Created on        :
% # last modified     :   12/12/2019 Thu
% # Description       :
% # 2. RGB
%   (finally used in paper) bler performance 
%   constellation comparisions (constellation design vs. AE)
%===============================================================================
clc
clear all;
% close all;
%===============================================================================
% parameters
%===============================================================================
random_symbol_num_per_color=1e6;
color_num = 3;
SNR_dB     =   [0 : 2 : 20];
% M = 2; % modulation order
xi = 0.1;
InterferenceMatrix = [(1-xi)    xi             0
    xi      (1-2*xi)      xi
    0        xi          (1-xi)];
H = InterferenceMatrix;% * diag(1./[0.0114; 0.0052; 0.0427]);

H_noise_std = 0;
H_noise = H_noise_std * randn( color_num,color_num,random_symbol_num_per_color );
ber_all=[];
s_opt_all = zeros(3,8,3);
for i_color = [1 ]
    %******************************************
    if i_color ==1
        x0=0.313;y0=0.337;           %6500kcenter point
        g11=86e4;g12=-40e4;g22=45e4;%????
    elseif  i_color==2
        x0=0.346;y0=0.359;           %5000kcenter point
        g11=56e4;g12=-25e4;g22=28e4;%????
    elseif  i_color==3
        x0=0.459;y0=0.412;           %2700kcenter point
        g11=40e4;g12=-19.5e4;g22=28e4;%????
    end
    ksi=7;
    alpha=sqrt(2/((g11+g22)-sqrt((g11-g22)^2+(2*g12)^2)));
    beta=sqrt(2/((g11+g22)+sqrt((g11-g22)^2+(2*g12)^2)));
    if g12==0&&g11<g22
        theta=0;
    else if g12==0&&g11>g22
            theta=pi/2;
        else if g12~=0&&g11<g22
                theta=0.5*(cot((g11-g22)/(2*g12)))^-1;
            else if g12~=0&&g11>g22
                    theta=pi/2+0.5*(cot((g11-g22)/(2*g12)))^-1;
                end
            end
        end
    end
    x_old = [0.7006;0.1547;0.1440];
    y_old=[0.2993;0.8059;0.0297];
    % x_old = [0.69406;0.22965;0.12301];
    % y_old=[0.30257;0.70992;0.09249];
    a=x_old./y_old;
    b=1./y_old;
    
    %%%%%%%%%%%%%%%%%
    M=8;      %????
    c=[0.0114;0.0052;0.0427];      %RGB???????????
    I_max=[1;1;1];           %LED??????
    %%
    %%%%%??????????s0%%%%%%%%%%%%%%%%%
    %??????
    Xr=0.7006;Yr=0.2993;
    Xg=0.1547;Yg=0.8059;
    Xb=0.1440;Yb=0.0297;
    A1=[Xr/Yr,Xg/Yg,Xb/Yb;1,1,1;(1-Xr-Yr)/Yr,(1-Xg-Yg)/Yg,(1-Xb-Yb)/Yb];
    b_0=[x0/y0;1;(1-x0-y0)/y0];
    % ??????????
    ratio_0=(A1^-1)*b_0;       %??????????
    %%
    Dmax=0;
    S1=0:I_max(1)/(M-1):I_max(1);
    S2=0:I_max(1)/(M-1):I_max(1);
    S3=0:I_max(1)/(M-1):I_max(1);
    s_opt = [S1 ; S2 ;S3];
    for loop=1:10   %????100???
        %%
        %             S1=0:I_max(1)/(M-1):I_max(1);
        %             S2=0:I_max(1)/(M-1):I_max(1);
        %             S3=0:I_max(1)/(M-1):I_max(1);
        %             S = [S1 ; S2 ;S3];
        S=unifrnd(0+0.0001,I_max(1)-0.0001,3,M);
        s0=S;
        %%
        l=combntns(1:M,2); %???
        row=size(l,1); %???
        %%
        %%*************CVX**********************%
        j=0;
        D_old = 0;
        while j<=10   %??????10?
            i_color
            loop
            j
            cvx_begin
            variables fai(3) s(3,M);
            variables t D;
            maximize D
            subject to
            for i=1:row
                2 * ( s0(:,l(i,1))-s0(:,l(i,2)) )'* H'* ...
                    H * ( s (:,l(i,1))-s (:,l(i,2)) ) - ...
                    ( s0(:,l(i,1))-s0(:,l(i,2)) )'*  H'* ...
                    H * ( s0(:,l(i,1))-s0(:,l(i,2)) ) >=D^2;
            end
            
            m=1/alpha*((a'-x0*b')*cos(theta)+([1 1 1]-y0*b')*sin(theta))*fai;
            n=1/beta*(([1 1 1]-y0*b')*cos(theta)-(a'-x0*b')*sin(theta))*fai;
            0<=t<=ksi.*b'*fai;
            {[m;n],t} <In> lorentz(2);
            fai>=0;
            %         [1 1 1]*fai==Lt;
            sum(s,2)/M == fai.*c;
            0<=s<=I_max(1);
            cvx_end
            %         for kk=1:28
            %             aaaa(:,kk)=2*s0'* E1(:,:,kk)'* E1(:,:,kk)*s-s0'* E1(:,:,kk)'* E1(:,:,kk)*s0;
            %         end
            %         sqrt(min(aaaa))
            j=j+1;
            s0=s;
            if(norm((D - D_old)/sqrt(D^2+D_old^2),1) <= 0.001) || isnan(m) || isnan(n)
                break;
            end
            D_old = D;
        end
        %%
        %?100???????????
        if Dmax/sqrt(mean(mean((H*s_opt - mean(mean(H*s_opt))).^2))) <= D/sqrt(mean(mean((H*s - mean(mean(H*s))).^2)))
            Dmax=D;
            s_opt=s;
            fai_opt=fai;
        end
    end
    s = reshape(s_opt,3,M);
    s_opt_all(:,:,i_color) = s;
    current_mean = mean(s,2);
    current_mean/current_mean(3)
    DD = 100;
    DD_vector = [];
    for i=1:row
        D_temp=norm(H*s(:,l(i,1))-H*s(:,l(i,2)));
        DD_vector = [DD_vector; D_temp];
        if D_temp<DD
            DD = D_temp;
        end
    end
    %===============================================================================
    % ber caculation
    %===============================================================================
    
    ber = zeros(1,length(SNR_dB));
    for i_p_tx=1:length(SNR_dB)
        SNR = SNR_dB(i_p_tx);
        num_loop = 10;
        ber_sum = 0;
        for mm = 1:num_loop
            %===============================================================================
            % gen random bits
            %===============================================================================
            u_data = randi([1 M],1,random_symbol_num_per_color);
            for i_random_num = 1:random_symbol_num_per_color
                modData(:,i_random_num) = s(:,u_data(i_random_num));
            end
            rec_data1 = H * modData;
            for i_random_num = 1:random_symbol_num_per_color
                rec_data2(:,i_random_num) = H_noise(:,:,i_random_num) * modData(:,i_random_num);
            end
            rec_data = rec_data1 + rec_data2;
            rec_data_mean = mean(mean(rec_data));
            rec_data = rec_data - rec_data_mean;
            rec_data_with_noise = awgn(rec_data,SNR,'measured'); % mean of signal must be 0
            %                     rec_data_with_noise = rec_data;
            rec_data_with_noise = rec_data_with_noise + rec_data_mean;
            
            for i_random_num = 1:random_symbol_num_per_color
                r = rec_data_with_noise(:,i_random_num);
                d000 = norm( H*s(:,1)-r ) ;
                d001 = norm( H*s(:,2)-r ) ;
                d010 = norm( H*s(:,3)-r ) ;
                d011 = norm( H*s(:,4)-r ) ;
                d100 = norm( H*s(:,5)-r ) ;
                d101 = norm( H*s(:,6)-r ) ;
                d110 = norm( H*s(:,7)-r ) ;
                d111 = norm( H*s(:,8)-r ) ;
                min_d=min([d000 d001 d010 d011 d100 d101 d110 d111 ]')';
                if(min_d==d000)
                    y=1;
                elseif(min_d==d001)
                    y=2;
                elseif(min_d==d010)
                    y=3;
                elseif(min_d==d011)
                    y=4;
                elseif(min_d==d100)
                    y=5;
                elseif(min_d==d101)
                    y=6;
                elseif(min_d==d110)
                    y=7;
                else
                    y=8;
                end
                ber_sum = ber_sum + ( y~=u_data(i_random_num));
            end
        end
        % bit error num / loop num / group num
        ber(i_p_tx)=ber_sum/random_symbol_num_per_color/num_loop;
        %**********************************************************************
    end
    ber_all = [ber_all;ber];
end
% bler results are used in MIMO-PAM comparisions
semilogy(SNR_dB, ber_all(1,:), 'b-o');
hold on;
grid on
semilogy(SNR_dB, ber_all(2,:), 'b-+');
semilogy(SNR_dB, ber_all(3,:), 'b--');
title('RGB')
ylim([1.0E-6,1]);
% legend('baseline 6500K','baseline 5000K','baseline 2700K','AE 6500K','AE 5000K','AE 2700K')
save('RGB_same_h_constellation_design_vs_AE.mat')

%%
% constellation comparision with AE
% 6500K
s_opt_all(:,:,1)=[    0.0000    1.0000    0.0000    0.6918    0.6843    1.0000    0.3433    0.3403;
    1.0000    0.2985    0.3215    0.0000    1.0000    1.0000    0.2376    0.9025;
    0.5840    0.4529    0.0000    0.0000    0.6120    0.0000    0.6116    0.0000];
s_opt_all(:,:,1)=roundn(s_opt_all(:,:,1),-2);
s_ae_all(:,:,1)=[9.4369404e-02 6.4517386e-02 2.1498092e-02
    4.5200017e-01 1.4356887e-02 4.6603414e-01
    1.8881721e-02 6.2057507e-01 5.9118081e-04
    2.8837940e-01 1.0000000e+00 5.5779330e-04
    7.2830141e-01 5.5293334e-01 5.2397046e-04
    4.3834090e-02 4.9686348e-01 4.8555976e-01
    6.0941255e-01 5.7986777e-02 1.6458471e-03
    5.4967028e-01 5.7992244e-01 4.5953488e-01];
s_ae_all(:,:,1)=roundn(s_ae_all(:,:,1),-4);
figure;
%CCK constellation
scatter3(s_opt_all(1,:,1),s_opt_all(2,:,1),s_opt_all(3,:,1),'MarkerEdgeColor','k','MarkerFaceColor','c')
for i_constellation = 1:8
    s=[' (' num2str(s_opt_all(1,i_constellation,1)) ','  num2str((s_opt_all(2,i_constellation,1))) ',' num2str((s_opt_all(3,i_constellation,1))) ')'];
    text(s_opt_all(1,i_constellation,1),s_opt_all(2,i_constellation,1),s_opt_all(3,i_constellation,1)-0.03, s, 'fontsize', 10);
end
hold on;
%AE constellation
scatter3(s_ae_all(1,:,1),s_ae_all(2,:,1),s_ae_all(3,:,1),'MarkerEdgeColor','k','MarkerFaceColor','r')
for i_constellation = 1:8
    s=[' (' num2str(s_ae_all(1,i_constellation,1)) ','  num2str((s_ae_all(2,i_constellation,1))) ',' num2str((s_ae_all(3,i_constellation,1))) ')'];
    text(s_ae_all(1,i_constellation,1),s_ae_all(2,i_constellation,1),s_ae_all(3,i_constellation,1)+0.03, s, 'fontsize', 10);
end
%mean1 (CCK constellation)
s_ae_mean = mean(s_opt_all(:,:,1),2);
s_ae_mean = roundn(s_ae_mean,-2);
scatter3(s_ae_mean(1),s_ae_mean(2),s_ae_mean(3),'d','MarkerEdgeColor','k','MarkerFaceColor','w')
s=['mean 1: (' num2str(s_ae_mean(1)) ','  num2str((s_ae_mean(2))) ',' num2str((s_ae_mean(3))) ')'];
text(s_ae_mean(1),s_ae_mean(2),s_ae_mean(3)+0.03, s, 'fontsize', 10);
%mean2 (AE constellation)
s_ae_mean = mean(s_ae_all(:,:,1),2);
s_ae_mean = roundn(s_ae_mean,-2);
scatter3(s_ae_mean(1),s_ae_mean(2),s_ae_mean(3),'s','MarkerEdgeColor','k','MarkerFaceColor','w')
s=['mean 2: (' num2str(s_ae_mean(1)) ','  num2str((s_ae_mean(2))) ',' num2str((s_ae_mean(3))) ')'];
text(s_ae_mean(1),s_ae_mean(2),s_ae_mean(3)+0.03, s, 'fontsize', 10);
%target
x0=0.313;y0=0.337;
b_0=[x0/y0;1;(1-x0-y0)/y0];
ratio_0=((A1^-1)*b_0).*c;
ratio_0 = roundn(ratio_0/ratio_0(1)*s_ae_mean(1),-2);

scatter3(ratio_0(1),ratio_0(2),ratio_0(3),'*','k')
s=['center: (' num2str(ratio_0(1)) ','  num2str((ratio_0(2))) ',' num2str((ratio_0(3))) ')'];
text(ratio_0(1),ratio_0(2),ratio_0(3)+0.03, s, 'fontsize', 10);

xlabel('red');ylabel('green');zlabel('blue');
legend('constellations points (baseline 2)','constellations points (proposed scheme)',...
    'mean value 1 (baseline 2)','mean value 2 (proposed scheme)',...
    'center point of the MacAdam ellipse','Location','northeast');
% title('6500K');
%MED
DD_opt_6500 = 100;
for i=1:row
    D_temp=norm(H*s_opt_all(:,l(i,1),1)-H*s_opt_all(:,l(i,2),1))...
        /sqrt(mean(mean( (H*s_opt_all(:,:,1)-mean(mean(H*s_opt_all(:,:,1)))).^2,2)));
    if D_temp<DD_opt_6500
        DD_opt_6500 = D_temp;
    end
end
DD_ae_6500 = 100;
for i=1:row
    D_temp=norm(H*s_ae_all(:,l(i,1),1)-H*s_ae_all(:,l(i,2),1))...
        /sqrt(mean(mean( (H*s_ae_all(:,:,1)-mean(mean(H*s_ae_all(:,:,1)))).^2,2)));
    if D_temp<DD_ae_6500
        DD_ae_6500 = D_temp;
    end
end
%%
% 5000K
s_opt_all(:,:,2)=[    0.0000    0.0000    0.3314    1.0000    0.5592    0.6408    0.5515    1.0000;
    0.2995    1.0000    0.0165    0.0000    1.0000    0.0602    0.6155    1.0000;
    0.4548    0.1756    0.0000    0.0000    0.4497    0.5254    0.0000    0.0000];
s_opt_all(:,:,2)=roundn(s_opt_all(:,:,2),-2);
s_ae_all(:,:,2)=[0.09328018 1.         0.00186636
    0.758483   0.7682035  0.00109632
    0.21392012 0.00513195 0.00144293
    0.46660182 0.65364337 0.45474702
    0.38730305 0.4722089  0.0020229
    0.5439526  0.01789392 0.4603642
    0.00454329 0.4276297  0.29319197
    0.8422292  0.09963633 0.00277831];
s_ae_all(:,:,2)=roundn(s_ae_all(:,:,2),-4);
figure;
%CCK constellation
scatter3(s_opt_all(1,:,2),s_opt_all(2,:,2),s_opt_all(3,:,2),'MarkerEdgeColor','k','MarkerFaceColor','c')
for i_constellation = 1:8
    s=[' (' num2str(s_opt_all(1,i_constellation,2)) ','  num2str((s_opt_all(2,i_constellation,2))) ',' num2str((s_opt_all(3,i_constellation,2))) ')'];
    text(s_opt_all(1,i_constellation,2),s_opt_all(2,i_constellation,2),s_opt_all(3,i_constellation,2)-0.03, s, 'fontsize', 10);
end
hold on;
%AE constellation
scatter3(s_ae_all(1,:,2),s_ae_all(2,:,2),s_ae_all(3,:,2),'MarkerEdgeColor','k','MarkerFaceColor','r')
for i_constellation = 1:8
    s=[' (' num2str(s_ae_all(1,i_constellation,2)) ','  num2str((s_ae_all(2,i_constellation,2))) ',' num2str((s_ae_all(3,i_constellation,2))) ')'];
    text(s_ae_all(1,i_constellation,2),s_ae_all(2,i_constellation,2),s_ae_all(3,i_constellation,2)+0.03, s, 'fontsize', 10);
end

%mean1 (CCK constellation)
s_ae_mean = mean(s_opt_all(:,:,2),2);
s_ae_mean = roundn(s_ae_mean,-2);
scatter3(s_ae_mean(1),s_ae_mean(2),s_ae_mean(3),'d','MarkerEdgeColor','k','MarkerFaceColor','w')
s=['mean 1: (' num2str(s_ae_mean(1)) ','  num2str((s_ae_mean(2))) ',' num2str((s_ae_mean(3))) ')'];
text(s_ae_mean(1),s_ae_mean(2),s_ae_mean(3)+0.03, s, 'fontsize', 10);
%mean (AE constellation)
s_ae_mean = mean(s_ae_all(:,:,2),2);
s_ae_mean = roundn(s_ae_mean,-2);
scatter3(s_ae_mean(1),s_ae_mean(2),s_ae_mean(3),'s','MarkerEdgeColor','k','MarkerFaceColor','w')
s=['mean 2: (' num2str(s_ae_mean(1)) ','  num2str((s_ae_mean(2))) ',' num2str((s_ae_mean(3))) ')'];
text(s_ae_mean(1),s_ae_mean(2),s_ae_mean(3)+0.03, s, 'fontsize', 10);

%target
x0=0.346;y0=0.359;
b_0=[x0/y0;1;(1-x0-y0)/y0];
ratio_0=((A1^-1)*b_0).*c;
ratio_0 = roundn(ratio_0/ratio_0(1)*s_ae_mean(1),-2);

scatter3(ratio_0(1),ratio_0(2),ratio_0(3),'*','k')
s=['center: (' num2str(ratio_0(1)) ','  num2str((ratio_0(2))) ',' num2str((ratio_0(3))) ')'];
text(ratio_0(1),ratio_0(2),ratio_0(3)+0.03, s, 'fontsize', 10);

xlabel('red');ylabel('green');zlabel('blue');
legend('constellations points (baseline 2)','constellations points (proposed scheme)',...
    'mean value 1 (baseline 2)','mean value 2 (proposed scheme)',...
    'center point of the MacAdam ellipse','Location','northeast');
% title('5000K');
DD_opt_5000 = 100;
for i=1:row
    D_temp=norm(H*s_opt_all(:,l(i,1),2)-H*s_opt_all(:,l(i,2),2))...
        /sqrt(mean(mean( (H*s_opt_all(:,:,2)-mean(mean(H*s_opt_all(:,:,2)))).^2,2)));
    if D_temp<DD_opt_5000
        DD_opt_5000 = D_temp;
    end
end
DD_ae_5000 = 100;
for i=1:row
    D_temp=norm(H*s_ae_all(:,l(i,1),2)-H*s_ae_all(:,l(i,2),2))...
        /sqrt(mean(mean( (H*s_ae_all(:,:,2)-mean(mean(H*s_ae_all(:,:,2)))).^2,2)));
    if D_temp<DD_ae_5000
        DD_ae_5000 = D_temp;
    end
end
%%
% 2700K
s_opt_all(:,:,3)=[    1.0000    0.1481    1.0000    0.5031    0.2743    0.7447    0.6449    0.0000;
    0.0276    0.4992    0.8431    0.0000    1.0000    0.0000    0.5268    0.0000;
    0.0000    0.0000    0.0000    0.0000    0.0369    0.4384    0.0000    0.0000];
s_opt_all(:,:,3)=roundn(s_opt_all(:,:,3),-2);
s_ae_all(:,:,3)=[1.00000000e+00 5.88326948e-04 1.67359860e-04
    7.40186637e-03 5.39488673e-01 5.31365396e-04
    4.00855958e-01 2.62525654e-03 3.83722842e-01
    2.19819741e-03 1.11290392e-04 2.77845229e-05
    9.71229792e-01 5.66762030e-01 3.21512838e-04
    5.05321443e-01 2.32023609e-04 1.02957216e-04
    5.47087193e-01 4.23587203e-01 1.27521605e-04
    3.50894004e-01 9.23743248e-01 4.42420103e-04];
s_ae_all(:,:,3)=roundn(s_ae_all(:,:,3),-2);
figure;
%CCK constellation
scatter3(s_opt_all(1,:,3),s_opt_all(2,:,3),s_opt_all(3,:,3),'MarkerEdgeColor','k','MarkerFaceColor','c')
for i_constellation = 1:8
    s=[' (' num2str(s_opt_all(1,i_constellation,3)) ','  num2str((s_opt_all(2,i_constellation,3))) ',' num2str((s_opt_all(3,i_constellation,3))) ')'];
    text(s_opt_all(1,i_constellation,3),s_opt_all(2,i_constellation,3),s_opt_all(3,i_constellation,3)+0.03, s, 'fontsize', 10);
end
hold on;
%AE constellation
scatter3(s_ae_all(1,:,3),s_ae_all(2,:,3),s_ae_all(3,:,3),'MarkerEdgeColor','k','MarkerFaceColor','r')
for i_constellation = 1:8
    s=[' (' num2str(s_ae_all(1,i_constellation,3)) ','  num2str((s_ae_all(2,i_constellation,3))) ',' num2str((s_ae_all(3,i_constellation,3))) ')'];
    text(s_ae_all(1,i_constellation,3),s_ae_all(2,i_constellation,3),s_ae_all(3,i_constellation,3)-0.03, s, 'fontsize', 10);
end

%mean1 (CCK constellation)
s_ae_mean = mean(s_opt_all(:,:,3),2);
s_ae_mean = roundn(s_ae_mean,-2);
scatter3(s_ae_mean(1),s_ae_mean(2),s_ae_mean(3),'d','MarkerEdgeColor','k','MarkerFaceColor','w')
s=['mean 1: (' num2str(s_ae_mean(1)) ','  num2str((s_ae_mean(2))) ',' num2str((s_ae_mean(3))) ')'];
text(s_ae_mean(1),s_ae_mean(2),s_ae_mean(3)+0.03, s, 'fontsize', 10);
%mean (AE constellation)
s_ae_mean = mean(s_ae_all(:,:,3),2);
s_ae_mean = roundn(s_ae_mean,-2);
scatter3(s_ae_mean(1),s_ae_mean(2),s_ae_mean(3),'s','MarkerEdgeColor','k','MarkerFaceColor','w')
s=['mean 2: (' num2str(s_ae_mean(1)) ','  num2str((s_ae_mean(2))) ',' num2str((s_ae_mean(3))) ')'];
text(s_ae_mean(1)-0.1,s_ae_mean(2),s_ae_mean(3), s, 'fontsize', 10);
%target
x0=0.459;y0=0.412;
b_0=[x0/y0;1;(1-x0-y0)/y0];
ratio_0=((A1^-1)*b_0).*c;
ratio_0 = roundn(ratio_0/ratio_0(1)*s_ae_mean(1),-2);

scatter3(ratio_0(1),ratio_0(2),ratio_0(3),'*','k')
s=['center: (' num2str(ratio_0(1)) ','  num2str((ratio_0(2))) ',' num2str((ratio_0(3))) ')'];
text(ratio_0(1)+0.1,ratio_0(2),ratio_0(3), s, 'fontsize', 10);

xlabel('red');ylabel('green');zlabel('blue');
legend('constellations points (baseline 2)','constellations points (proposed scheme)',...
    'mean value 1 (baseline 2)','mean value 2 (proposed scheme)',...
    'center point of the MacAdam ellipse','Location','northeast');
% title('2700K');
DD_opt_2700 = 100;
for i=1:row
    D_temp=norm(H*s_opt_all(:,l(i,1),3)-H*s_opt_all(:,l(i,2),3))...
        /sqrt(mean(mean( (H*s_opt_all(:,:,3)-mean(mean(H*s_opt_all(:,:,3)))).^2,2)));
    if D_temp<DD_opt_2700
        DD_opt_2700 = D_temp;
    end
end
DD_ae_2700 = 100;
for i=1:row
    D_temp=norm(H*s_ae_all(:,l(i,1),3)-H*s_ae_all(:,l(i,2),3))...
        /sqrt(mean(mean( (H*s_ae_all(:,:,3)-mean(mean(H*s_ae_all(:,:,3)))).^2,2)));
    if D_temp<DD_ae_2700
        DD_ae_2700 = D_temp;
    end
end