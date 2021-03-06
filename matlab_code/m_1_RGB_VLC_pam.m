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
% H	=    ChannelMatrix_2T_2R;
%has little effect on performance
H = InterferenceMatrix * diag([1.1408 ; 1.3226 ; 1]./[0.0114; 0.0052; 0.0427]);

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
%         modData_tx_1=(modData_tx).*(modData_tx<=0.5)+0.5.*(modData_tx>0.5); % clipping
%         modData_tx_2=(modData_tx_1).*(modData_tx_1>=-0.5)-0.5.*(modData_tx_1<-0.5); % clipping
        %1.1408 : 1.3226 : 1 (I DC,i,R : I DC,i,G : I DC,i,B )
%         rgb_propotion = [1.1408 ; 1.3226 ; 1]./[0.0114; 0.0052; 0.0427];
%         rgb_rep_matrix = repmat(rgb_propotion,1,random_symbol_num_per_color);
        
%         tx_data = rgb_rep_matrix .* modData;
        tx_data = modData;
        
        
        rec_data = H * tx_data;
        rec_data_normalize = rec_data/sqrt(mean(mean(rec_data.^2)));
        rec_data_with_noise = rec_data_normalize + sqrt(P_noise)*randn(size(rec_data));
        %ZF receiver
        rec_data_with_noise = pinv(H) *rec_data_with_noise;
        rec_data_with_noise = rec_data_with_noise;
        rec_data_with_noise_normalize = rec_data_with_noise/sqrt(mean(mean(rec_data_with_noise.^2)));
        rec_dec = pamdemod(rec_data_with_noise_normalize*sqrt(mean(mean(modData.^2))),M);

        ber_temp = symerr(u_data,rec_dec)/color_num/random_symbol_num_per_color;
        ber_sum = ber_sum + ber_temp;
    end
    % bit error num / loop num / group num
    ber(i_p_tx)=ber_sum/num_loop;
    %**********************************************************************
end
figure;
semilogy(SNR_dB, ber, 'b-o');
hold on;
grid on



