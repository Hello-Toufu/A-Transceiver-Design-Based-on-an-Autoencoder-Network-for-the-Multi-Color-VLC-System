clear;clc;
%===============================================================================
% parameters
%===============================================================================
random_symbol_num_per_color=1e5;
color_num = 3;
SNR_dB     =   [0 : 5 : 25];
psi_2 = 5;
M = 2; % modulation order
xi = 0.1;
InterferenceMatrix = [(1-xi)    xi             0
    xi      (1-2*xi)      xi
    0        xi          (1-xi)];


%??????
Xr=0.7006;Yr=0.2993;
Xg=0.1547;Yg=0.8059;
Xb=0.1440;Yb=0.0297;
A1=[Xr/Yr,Xg/Yg,Xb/Yb;1,1,1;(1-Xr-Yr)/Yr,(1-Xg-Yg)/Yg,(1-Xb-Yb)/Yb];
c=[0.0114;0.0052;0.0427];
%6500K [1.1408 ; 1.3226 ; 1]
%5700 [1.3284   ; 1.3912 ;   1.0000]
%5000K [1.6339 ;   1.6348 ;   1.0000]
%4000 [2.3762  ;  2.0844  ;  1.0000]
%2700 [6.1892  ;  3.9912  ;  1.0000]
        x0=0.313;y0=0.337;
        b_0=[x0/y0;1;(1-x0-y0)/y0];
        % ?????????? %??????????
        ratio_0=(A1^-1)*b_0;      
        H = InterferenceMatrix * diag(ratio_0.*c);
%===============================================================================
% ber caculation
%===============================================================================
ber_all=[];
for i=[1 2 3]
    %******************************************
    if i ==1
        psi_2 = 1;


    elseif  i==2
        psi_2 = 2;
    elseif  i==3
psi_2 = 5;
    end
    %****************parameter init*************************
    % H = InterferenceMatrix * diag([1.1408 ; 1.3226 ; 1]./[0.0114; 0.0052; 0.0427]);
    
    %initialize
    ber = zeros(1,length(SNR_dB));
    for i_p_tx=1:length(SNR_dB)
        SNR = 10.0 ^ (SNR_dB(i_p_tx) / 10.0);
        P_noise = 1/SNR;
        
        num_loop = 10;
        ber_sum = 0;
        for mm = 1:num_loop
            %===============================================================================
            % gen random bits
            %===============================================================================
            u_data = randi([0 M-1],color_num,random_symbol_num_per_color);
            modData = real(pammod(u_data,M)) + 1;
            
            rec_data = H * modData;
            rec_data_mean = mean(mean(rec_data));
            rec_data_power = (mean(mean((rec_data-rec_data_mean).^2)));
            
%             rec_data_normalize = rec_data/rec_data_std;
%             rec_data_with_noise = rec_data_normalize + sqrt(P_noise)*randn(size(rec_data));
            rec_data_with_noise = rec_data + sqrt(P_noise*rec_data_power*(1+psi_2*rec_data)).*randn(size(rec_data));
            
%             rec_data_with_noise = rec_data_with_noise*rec_data_std - 1;
            rec_data_with_noise = rec_data_with_noise - 1;
            for t = 1:random_symbol_num_per_color
                r = rec_data_with_noise(:,t);
                d000 = norm( H*real(pammod([0;0;0],M))-r ) ;
                d001 = norm( H*real(pammod([0;0;1],M))-r ) ;
                d010 = norm( H*real(pammod([0;1;0],M))-r ) ;
                d011 = norm( H*real(pammod([0;1;1],M))-r ) ;
                d100 = norm( H*real(pammod([1;0;0],M))-r ) ;
                d101 = norm( H*real(pammod([1;0;1],M))-r ) ;
                d110 = norm( H*real(pammod([1;1;0],M))-r ) ;
                d111 = norm( H*real(pammod([1;1;1],M))-r ) ;
                min_d=min([d000 d001 d010 d011 d100 d101 d110 d111 ]')';
                if(min_d==d000)
                    y=[0;0;0];
                elseif(min_d==d001)
                    y=[0;0;1];
                elseif(min_d==d010)
                    y=[0;1;0];
                elseif(min_d==d011)
                    y=[0;1;1];
                elseif(min_d==d100)
                    y=[1;0;0];
                elseif(min_d==d101)
                    y=[1;0;1];
                elseif(min_d==d110)
                    y=[1;1;0];
                else
                    y=[1;1;1];
                end
                ber_sum = ber_sum + (sum( y~=u_data(:,t))>0);
            end
        end
        % bit error num / loop num / group num
        ber(i_p_tx)=ber_sum/random_symbol_num_per_color/num_loop;
        %**********************************************************************
    end
    ber_all = [ber_all;ber];
end
% figure;
semilogy(SNR_dB, ber_all(1,:), 'b-o');
hold on;
grid on
% semilogy(SNR_dB, ber_all(2,:), 'b-*');
semilogy(SNR_dB, ber_all(2,:), 'b-+');
% semilogy(SNR_dB, ber_all(4,:), 'b--');
semilogy(SNR_dB, ber_all(3,:), 'b--');

semilogy(SNR_dB, [0.62239999 0.38589999 0.0935     0.0025     0.         0.        ], 'r-o');
% semilogy(SNR_dB, [0.61619997 0.3827     0.111      0.0047     0.        ], 'r-*');
semilogy(SNR_dB, [0.63520002 0.40349999 0.1105     0.0027     0.         0.        ], 'r-+');
semilogy(SNR_dB, [0.66219997 0.4596     0.15260001 0.0095     0.         0.        ], 'r--');

semilogy(SNR_dB, [0.44010001 0.13699999 0.0049     0.           0], 'r-o');
semilogy(SNR_dB, [0.45899999 0.1584     0.0076     0.           0], 'r-+');
semilogy(SNR_dB, [0.51730001 0.24590001 0.0316     0.0006       0], 'r--');
title('RGB')

legend('baseline 6500K','baseline 5000K','baseline 2700K','AE 6500K','AE 5000K','AE 2700K')
save('RGB_same_h_shot_noise.mat')
