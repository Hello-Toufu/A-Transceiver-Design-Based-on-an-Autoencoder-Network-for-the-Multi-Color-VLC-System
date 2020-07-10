%===============================================================================
% # Author            :   ZDF
% # Created on        :
% # last modified     :   12/12/2019 Thu
% # Description       :
% # 4. RGBY
%   (finally used in paper) bler performance 
%   constellation comparisions (constellation design vs. AE)
%===============================================================================
clc
clear all;
% close all;
%===============================================================================
% parameters
%===============================================================================
random_symbol_num_per_color=1e5;
color_num = 4;
SNR_dB     =   [0 : 2 : 20];
% M = 2; % modulation order
xi = 0.1;
InterferenceMatrix = [(1-xi)    xi             0           0
    xi      (1-2*xi)      xi          0
    0       xi          (1-2*xi)    xi
    0       0            xi          (1-xi)];
H = InterferenceMatrix;% * diag(1./[0.0114; 0.0052; 0.0427]);
ber_all=[];
for i_color = [1]
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
    x_old = [0.69406;0.59785;0.22965;0.12301];
    y_old = [0.30257;0.39951;0.70992;0.09249];
    % x_old = [0.69406;0.22965;0.12301];
    % y_old=[0.30257;0.70992;0.09249];
    a=x_old./y_old;
    b=1./y_old;
    
    %%%%%%%%%%%%%%%%%
    M=16;      %????
    c=[0.021;0.014;0.005;0.015];      %RGB???????????
    I_max=[1;1;1;1];           %LED??????
    %%
    Dmax=0;
    S1=0:I_max(1)/(M-1):I_max(1);
    S2=0:I_max(1)/(M-1):I_max(1);
    S3=0:I_max(1)/(M-1):I_max(1);
    S4=0:I_max(1)/(M-1):I_max(1);
    s_opt = [S1 ; S2 ;S3 ; S4];
    for loop=1:10   %????100???
        %%
        %             S1=0:I_max(1)/(M-1):I_max(1);
        %             S2=0:I_max(1)/(M-1):I_max(1);
        %             S3=0:I_max(1)/(M-1):I_max(1);
        %             S = [S1 ; S2 ;S3];
        S=unifrnd(0+0.0001,I_max(1)-0.0001,4,M);
        s0=S;
        %%
        l=combntns(1:M,2); %???
        row=size(l,1); %???
        %%*************CVX**********************%
        j=0;
        D_old = 0;
        while j<=10   %??????10?
            i_color
            loop
            j
            cvx_begin
            variables fai(4) s(4,M);
            variables t D;
            maximize D
            subject to
            for i=1:row
                2 * ( s0(:,l(i,1))-s0(:,l(i,2)) )'* H'* ...
                    H * ( s (:,l(i,1))-s (:,l(i,2)) ) - ...
                    ( s0(:,l(i,1))-s0(:,l(i,2)) )'*  H'* ...
                    H * ( s0(:,l(i,1))-s0(:,l(i,2)) ) >=D^2;
            end
            
            m=1/alpha*((a'-x0*b')*cos(theta)+([1 1 1 1]-y0*b')*sin(theta))*fai;
            n=1/beta*(([1 1 1 1]-y0*b')*cos(theta)-(a'-x0*b')*sin(theta))*fai;
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
        
        %?100???????????
        if Dmax/sqrt(mean(mean((H*s_opt - mean(mean(H*s_opt))).^2))) <= D/sqrt(mean(mean((H*s - mean(mean(H*s))).^2)))
            Dmax=D;
            %             if mean(mean((s_opt - mean(mean(s_opt))).^2)) < mean(mean((s - mean(mean(s))).^2))
            s_opt=s;
            %             end
            fai_opt=fai;
        end
    end
    s = reshape(s_opt,4,M);
    current_mean = mean(s,2);
    DD = 100;
    DD_vector = [];
    for i=1:row
        D_temp=norm(H*s(:,l(i,1))-H*s(:,l(i,2)))/sqrt(mean(mean( (H*s).^2,2)));
        DD_vector = [DD_vector; D_temp];
        if D_temp<DD
            DD = D_temp;
        end
    end
    
    if i_color ==1
        DD_6500 = DD;
    elseif  i_color==2
        DD_5000 = DD;
    elseif  i_color==3
        DD_2700 = DD;
    end    %===============================================================================
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
            rec_data = H * modData;
%             rec_data_mean = mean(mean(rec_data));
%             rec_data = rec_data - rec_data_mean;
            rec_data_with_noise = awgn(rec_data,SNR,'measured'); % mean of signal must be 0
            %                     rec_data_with_noise = rec_data;
%             rec_data_with_noise = rec_data_with_noise + rec_data_mean;
            
            for i_random_num = 1:random_symbol_num_per_color
                r = rec_data_with_noise(:,i_random_num);
                d0000 = norm( H*s(:,1)-r ) ;
                d0001 = norm( H*s(:,2)-r ) ;
                d0010 = norm( H*s(:,3)-r ) ;
                d0011 = norm( H*s(:,4)-r ) ;
                d0100 = norm( H*s(:,5)-r ) ;
                d0101 = norm( H*s(:,6)-r ) ;
                d0110 = norm( H*s(:,7)-r ) ;
                d0111 = norm( H*s(:,8)-r ) ;
                d1000 = norm( H*s(:,9)-r ) ;
                d1001 = norm( H*s(:,10)-r ) ;
                d1010 = norm( H*s(:,11)-r ) ;
                d1011 = norm( H*s(:,12)-r ) ;
                d1100 = norm( H*s(:,13)-r ) ;
                d1101 = norm( H*s(:,14)-r ) ;
                d1110 = norm( H*s(:,15)-r ) ;
                d1111 = norm( H*s(:,16)-r ) ;
                [min_d,y]=min([d0000 d0001 d0010 d0011 d0100 d0101 d0110 d0111 ...
                    d1000 d1001 d1010 d1011 d1100 d1101 d1110 d1111]);
                ber_sum = ber_sum + ( y~=u_data(i_random_num));
            end
        end
        % bit error num / loop num / group num
        ber(i_p_tx)=ber_sum/random_symbol_num_per_color/num_loop;
        %**********************************************************************
    end
    ber_all = [ber_all;ber];
end
semilogy(SNR_dB, ber_all(1,:), 'b-o');
hold on;
grid on
semilogy(SNR_dB, ber_all(2,:), 'b-+');
semilogy(SNR_dB, ber_all(3,:), 'b--');


s_ae_all(:,:,1)=[3.6046809e-01 1.5445134e-02 4.9196947e-03 4.4032508e-01

 2.0724849e-01 4.5061585e-01 1.2423678e-03 1.2704584e-02

 4.0137570e-02 9.7243458e-02 5.1883662e-01 1.0475399e-02

 5.0014800e-01 4.1419375e-03 8.7214231e-01 4.6259549e-02

 5.8942044e-01 4.8910731e-01 6.6733462e-01 5.5736208e-01

 5.4373372e-01 8.0693328e-01 1.3089052e-01 2.1783058e-03

 1.2294092e-02 7.8145558e-01 4.0479356e-01 1.0119576e-03

 2.3038786e-02 5.7823879e-01 6.9660105e-02 5.5468708e-01

 8.9217925e-01 3.6481407e-01 5.8260834e-01 9.8403497e-04

 2.8142753e-01 1.0000000e+00 2.6787692e-01 4.6463022e-01

 6.6106188e-01 2.7396092e-02 2.5691074e-01 2.0619852e-03

 8.4726059e-01 2.6169366e-03 4.3988240e-01 4.7952509e-01

 2.9863611e-01 1.1445928e-02 5.5653197e-01 5.8782625e-01

 2.9271448e-02 4.9265262e-01 7.6820379e-01 3.6384606e-01

 4.2763114e-01 6.5981144e-01 8.3017111e-01 1.0409359e-03

 6.2195808e-01 5.3369677e-01 1.0836379e-02 5.1571244e-01]';
DD_ae_6500 = 100;
for i=1:row
    D_temp=norm(H*s_ae_all(:,l(i,1),1)-H*s_ae_all(:,l(i,2),1))...
        /sqrt(mean(mean( (H*s_ae_all(:,:,1)-mean(mean(H*s_ae_all(:,:,1)))).^2,2)));
    if D_temp<DD_ae_6500
        DD_ae_6500 = D_temp;
    end
end


% semilogy(SNR_dB, [0.4993        0.1523      0.0028      0.      0], 'r-o');
% semilogy(SNR_dB, [0.52539998	0.1698      0.0045      0.      0], 'r-*');
% semilogy(SNR_dB, [0.59500003	0.25870001	0.0207      0.      0], 'r-+');
title('RGBY')
ylim([1.0E-5,1]);
legend('baseline (\sigma_e=0)','baseline (\sigma_e=0.05)','baseline (\sigma_e=0.1)',...
    'AE (\sigma_e=0)','AE (\sigma_e=0.05)','AE (\sigma_e=0.1)')
save('RGBY_same_h_constellation_design_vs_AE_multi_CCT.mat')

