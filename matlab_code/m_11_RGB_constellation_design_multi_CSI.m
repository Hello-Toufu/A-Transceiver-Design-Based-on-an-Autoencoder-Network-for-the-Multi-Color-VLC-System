%===============================================================================
% # Author            :   ZDF
% # Created on        :
% # last modified     :   12/18/2019 Thu
% # Description       :
% # 5. RGB
%   (finally used in paper) bler performance comparision : 
%	constellation design (ML) vs. AE
%   multi csi
%===============================================================================
clear;clc;
%===============================================================================
% parameters
%===============================================================================
random_symbol_num_per_color=1e5;
color_num = 3;
SNR_dB     =   [0 : 2 : 20];
M = 2; % modulation order
xi = 0.1;
InterferenceMatrix = [(1-xi)    xi             0
    xi      (1-2*xi)      xi
    0        xi          (1-xi)];
H = InterferenceMatrix;% * diag(1./[0.0114; 0.0052; 0.0427]);

x0=0.313;y0=0.337;           %6500kcenter point
g11=86e4;g12=-40e4;g22=45e4;%????
% H_noise_std = 0.05;
% H_noise = H_noise_std * randn( color_num,color_num,random_symbol_num_per_color );
ber_all=[];
for i_color = [1 2 3 ]
    %******************************************
    if i_color ==1
        H_noise_std = 0;
        H_noise = H_noise_std * randn( color_num,color_num,random_symbol_num_per_color );
    elseif  i_color==2
        H_noise_std = 0.05;
        H_noise = H_noise_std * randn( color_num,color_num,random_symbol_num_per_color );
    elseif  i_color==3
        H_noise_std = 0.1;
        H_noise = H_noise_std * randn( color_num,color_num,random_symbol_num_per_color );
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
%===============================================================================
% results visualization
%===============================================================================
figure;
semilogy(SNR_dB, ber_all(1,:), 'b--o');
hold on;
grid on;
semilogy(SNR_dB, ber_all(2,:), 'b--s');
semilogy(SNR_dB, ber_all(3,:), 'b--x');


semilogy(SNR_dB, [4.40722999e-01 3.20202002e-01 1.96746001e-01 9.26309995e-02...
 2.93199999e-02 5.10799997e-03 3.66000002e-04 5.99999985e-06...
 0.00000000e+00 0.00000000e+00 0.00000000e+00], 'r-o');
semilogy(SNR_dB, [4.36396995e-01 3.17341998e-01 1.94253001e-01 9.18759994e-02...
 2.91779999e-02 5.22300000e-03 4.19000001e-04 1.09999997e-05...
 0.00000000e+00 0.00000000e+00 0.00000000e+00], 'r-s');
semilogy(SNR_dB, [4.38599098e-01 3.24816000e-01 2.08319400e-01 1.08429000e-01...
 4.19871002e-02 1.07209000e-02 1.53460000e-03 9.36000004e-05...
 1.40000001e-06 0.00000000e+00 0.00000000e+00], 'r-x');
% title('RGB')
ylim([0.000006,1]);
legend('baseline 2 (\sigma_e=0)','baseline 2 (\sigma_e=0.05)','baseline 2 (\sigma_e=0.1)',...
    'proposed scheme (\sigma_e=0)','proposed scheme (\sigma_e=0.05)','proposed scheme (\sigma_e=0.1)')
xlabel('SNR (dB)')
ylabel('BLER')
% save workspace
save('RGB_same_h_constellation_design_vs_AE_multi_CSI.mat')

