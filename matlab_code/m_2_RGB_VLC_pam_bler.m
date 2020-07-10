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
% H = InterferenceMatrix * diag([1.1408 ; 1.3226 ; 1]./[0.0114; 0.0052; 0.0427]);
% H = InterferenceMatrix * diag([1,1,1]);
H = InterferenceMatrix * diag([1.1408 ; 1.3226 ; 1]);

%initialize
ber = zeros(1,length(SNR_dB));
for i_p_tx=1:length(SNR_dB)
	SNR = SNR_dB(i_p_tx);
% 	SNR = 10.0 ^ (SNR_dB(i_p_tx) / 10.0);
%     P_noise = 1/SNR;
    
    num_loop = 10;
    ber_sum = 0;
    for mm = 1:num_loop
        %===============================================================================
        % gen random bits
        %===============================================================================
        u_data = randi([0 M-1],color_num,random_symbol_num_per_color);
        modData = real(pammod(u_data,M));
        
        rec_data = H * modData;
        
        rec_data_with_noise = awgn(rec_data,SNR,'measured');
%         rec_data_normalize = rec_data/sqrt(mean(mean(rec_data.^2)));
%         rec_data_with_noise = rec_data_normalize + sqrt(P_noise)*randn(size(rec_data));
        %ZF receiver
        rec_data_with_noise = pinv(H) *rec_data_with_noise;
%         rec_data_with_noise_normalize = rec_data_with_noise;
%         rec_data_with_noise_normalize = (rec_data_with_noise-mean(mean(rec_data_with_noise)))/sqrt(mean(mean((rec_data_with_noise- mean(mean(rec_data_with_noise))).^2)));
        for i_color = 1 : color_num
            rec_data_with_noise_normalize(i_color,:) = rec_data_with_noise(i_color,:)-mean(rec_data_with_noise(i_color,:));
        end
        rec_dec = pamdemod(rec_data_with_noise_normalize*sqrt(mean(mean(modData.^2))),M);
        
        for dec_i = 1:random_symbol_num_per_color
            ber_sum = ber_sum + (sum( rec_dec(:,dec_i)~=u_data(:,dec_i))>0);
        end
    end
    % bit error num / loop num / group num
    ber(i_p_tx)=ber_sum/num_loop/random_symbol_num_per_color;
    %**********************************************************************
end
figure;
semilogy(SNR_dB, ber, 'b-o');
hold on;
grid on



