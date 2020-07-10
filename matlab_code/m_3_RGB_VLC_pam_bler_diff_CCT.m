clear;clc;
%===============================================================================
% parameters
%===============================================================================
random_symbol_num_per_color=1e5;
color_num = 3;
SNR_dB     =   [0 : 5 : 20];
M = 2; % modulation order
xi = 0.1;
InterferenceMatrix = [(1-xi)    xi             0
    xi      (1-2*xi)      xi
    0        xi          (1-xi)];

%6500K [1.1408 ; 1.3226 ; 1]
%5000K [1.6339 ;   1.6348 ;   1.0000]
%4000 [2.3762  ;  2.0844  ;  1.0000]
%2700 [6.1892  ;  3.9912  ;  1.0000]
ber_all=[];
for i=1:4
    %******************************************
    if i ==1
        H = InterferenceMatrix * diag([1.1408 ; 1.3226 ; 1]./[0.0114; 0.0052; 0.0427]);
    elseif  i==2
        H = InterferenceMatrix * diag([1.6339 ;   1.6348 ;   1.0000]./[0.0114; 0.0052; 0.0427]);
    elseif  i==3
        H = InterferenceMatrix * diag([2.3762  ;  2.0844  ;  1.0000]./[0.0114; 0.0052; 0.0427]);
    elseif  i==4
        H = InterferenceMatrix * diag([6.1892  ;  3.9912  ;  1.0000]./[0.0114; 0.0052; 0.0427]);
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
            modData = real(pammod(u_data,M));
            
            rec_data = H * modData;
            rec_data_normalize = rec_data/sqrt(mean(mean(rec_data.^2)));
            rec_data_with_noise = rec_data_normalize + sqrt(P_noise)*randn(size(rec_data));
            %ZF receiver
            rec_data_with_noise = pinv(H) *rec_data_with_noise;
            rec_data_with_noise_normalize = rec_data_with_noise/sqrt(mean(mean(rec_data_with_noise.^2)));
            rec_dec = pamdemod(rec_data_with_noise_normalize*sqrt(mean(mean(modData.^2))),M);
            
            for dec_i = 1:random_symbol_num_per_color
                ber_sum = ber_sum + (sum( rec_dec(:,dec_i)~=u_data(:,dec_i))>0);
            end
        end
        % bit error num / loop num / group num
        ber(i_p_tx)=ber_sum/num_loop/random_symbol_num_per_color;
        %**********************************************************************
    end
    ber_all = [ber_all;ber];
end
% figure;
semilogy(SNR_dB, ber_all(1,:), 'b-o');
hold on;
grid on
semilogy(SNR_dB, ber_all(2,:), 'b-*');
semilogy(SNR_dB, ber_all(3,:), 'b-+');
semilogy(SNR_dB, ber_all(4,:), 'b--');

semilogy(SNR_dB, [0.597      0.35980001 0.0914     0.0035     0.        ], 'r-o');
semilogy(SNR_dB, [0.61619997 0.3827     0.111      0.0047     0.        ], 'r-*');
semilogy(SNR_dB, [6.21200025e-01 3.97599995e-01 1.23199999e-01 6.59999996e-03 9.99999975e-05], 'r-+');
semilogy(SNR_dB, [0.64160001 0.44080001 0.1952     0.0403     0.0024    ], 'r--');
title('RGB')

legend('baseline 6500K','baseline 5000K','baseline 4000K','baseline 2700K','AE 6500K','AE 5000K','AE 4000K','AE 2700K')

