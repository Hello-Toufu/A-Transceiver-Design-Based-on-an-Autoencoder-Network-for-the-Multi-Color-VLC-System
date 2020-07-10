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
%5700 [1.3284   ; 1.3912 ;   1.0000]
%5000K [1.6339 ;   1.6348 ;   1.0000]
%4000 [2.3762  ;  2.0844  ;  1.0000]
%2700 [6.1892  ;  3.9912  ;  1.0000]
ber_all=[];
for i=[1 3 5]
    %******************************************
    if i ==1
        H = InterferenceMatrix * diag([1.1408 ; 1.3226 ; 1]./[0.0114; 0.0052; 0.0427]);
    elseif  i==2
        H = InterferenceMatrix * diag([1.3284   ; 1.3912 ;   1.0000]./[0.0114; 0.0052; 0.0427]);
    elseif  i==3
        H = InterferenceMatrix * diag([1.6339 ;   1.6348 ;   1.0000]./[0.0114; 0.0052; 0.0427]);
    elseif  i==4
        H = InterferenceMatrix * diag([2.3762  ;  2.0844  ;  1.0000]./[0.0114; 0.0052; 0.0427]);
    elseif  i==5
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
            
            rec_data_with_noise = rec_data_with_noise*sqrt(mean(mean(rec_data.^2)));
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

semilogy(SNR_dB, [0.54579997 0.28889999 0.052       0.0008     0.        ], 'r-o');
% semilogy(SNR_dB, [0.61619997 0.3827     0.111      0.0047     0.        ], 'r-*');
semilogy(SNR_dB, [0.55150002 0.28729999 0.0514      0.0008  0], 'r-+');
semilogy(SNR_dB, [0.55470002 0.29890001 0.0593     0.0009     0    ], 'r--');
title('RGB')

legend('baseline 6500K','baseline 5000K','baseline 4000K','baseline 2700K','AE 6500K','AE 5000K','AE 4000K','AE 2700K')
save('RGB.mat')
