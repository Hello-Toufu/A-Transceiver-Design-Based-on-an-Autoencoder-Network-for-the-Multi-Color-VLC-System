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

x0=0.313;y0=0.337;           %6500kcenter point
g11=86e4;g12=-40e4;g22=45e4;%????
% H_noise_std = 0.05;
% H_noise = H_noise_std * randn( color_num,color_num,random_symbol_num_per_color );
ber_all=[];
for i_color = [1 2 3]
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
%===============================================================================
% results visualization
%===============================================================================
figure;
semilogy(SNR_dB, ber_all(1,:), 'b--o');
hold on;
grid on;
semilogy(SNR_dB, ber_all(2,:), 'b--s');
semilogy(SNR_dB, ber_all(3,:), 'b--x');


semilogy(SNR_dB, [5.01893997e-01 3.56537998e-01 2.04539999e-01 8.35600004e-02...
 2.00529998e-02 2.03099999e-03 6.29999995e-05 9.99999975e-07...
 0.00000000e+00 0.00000000e+00 0.00000000e+00], 'r-o');
semilogy(SNR_dB, [5.01640996e-01 3.57358995e-01 2.07929000e-01 8.76779996e-02...
 2.29130000e-02 3.01600001e-03 1.45999999e-04 29.99999975e-07...
 0.00000000e+00 0.00000000e+00 0.00000000e+00], 'r-s');
semilogy(SNR_dB, [5.08732396e-01 3.74574900e-01 2.35166802e-01 1.17475001e-01...
 4.26270995e-02 1.00481001e-02 1.33020000e-03 7.05000006e-05...
 1.20000001e-06 0.00000000e+00 0.00000000e+00], 'r-x');
% title('RGB')
ylim([0.000006,1]);
legend('baseline 2 (\sigma_e=0)','baseline 2 (\sigma_e=0.05)','baseline 2 (\sigma_e=0.1)',...
    'proposed scheme (\sigma_e=0)','proposed scheme (\sigma_e=0.05)','proposed scheme (\sigma_e=0.1)')
xlabel('SNR (dB)')
ylabel('BLER')
% save workspace
save('RGBY_same_h_constellation_design_vs_AE_multi_CSI.mat')

