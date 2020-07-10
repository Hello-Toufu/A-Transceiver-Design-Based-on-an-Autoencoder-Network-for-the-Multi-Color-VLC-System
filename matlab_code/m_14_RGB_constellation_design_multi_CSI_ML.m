%===============================================================================
% # Author            :   ZDF
% # Created on        :
% # last modified     :   12/18/2019 Thu
% # Description       :
% # 5. RGB
%   (finally used in paper) bler performance comparision : 
%	constellation design (ML) vs. AE
%   multi csi
% % 14. LS estimation --> ML estimation
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
for i_color = [1  ]
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

%     s =[0.0000    1.0000    0.0000    0.6918    0.6843    1.0000    0.3433    0.3403;
%         1.0000    0.2985    0.3215    0.0000    1.0000    1.0000    0.2376    0.9025;
%         0.5840    0.4529    0.0000    0.0000    0.6120    0.0000    0.6116    0.0000];
s =[0.422063624389447,0.408451480391703,2.27182016101543e-10,0.549504848684201,0.644192888168630,0.999999999897098,4.30154711018541e-10,0.999999999902153;
    0.787004843870055,0.401328595962975,0.335129608565910,1.94026569668965e-10,0.999999999702871,0.999999999763847,0.999999999644408,0.276284888260660;
    1.16649462981805e-10,0.861896281453553,1.29695421039751e-10,0.393701171430365,0.547471986350923,1.14182400404534e-10,0.456680657493183,1.03572236012927e-10];
% s=[0.999999999788573,1.52193016023412e-10,0.538941059984677,0.125598406740447,0.856592869457364,2.60327482001531e-10,0.425840144139398,0.999999999873230;
%     0.244079953878526,0.538966036379729,0.0328980258163858,0.999999999669992,0.999999999795007,0.309665681205204,0.999999999919440,0.462089891743189;
%     6.47007373915740e-11,0.580349197218715,0.433397709852632,9.69000840486197e-11,1.41252329623714e-10,7.64440630730092e-11,0.583649180318133,0.585405694619308];
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
            
            awgn_noise_std = sqrt( mean(mean(rec_data.^2))/10^(SNR/10) ) ;
            %                     rec_data_with_noise = rec_data;
            rec_data_with_noise = rec_data_with_noise + rec_data_mean;
            
            for i_random_num = 1:random_symbol_num_per_color
                r = rec_data_with_noise(:,i_random_num);
                d000 =  norm(r-H*s(:,1))^2/(H_noise_std^2*norm(s(:,1))^2 + awgn_noise_std^2) + color_num*log( H_noise_std^2*norm(s(:,1))^2 + awgn_noise_std^2 ) ;
                d001 =  norm(r-H*s(:,2))^2/(H_noise_std^2*norm(s(:,2))^2 + awgn_noise_std^2) + color_num*log( H_noise_std^2*norm(s(:,2))^2 + awgn_noise_std^2 ) ;
                d010 =  norm(r-H*s(:,3))^2/(H_noise_std^2*norm(s(:,3))^2 + awgn_noise_std^2) + color_num*log( H_noise_std^2*norm(s(:,3))^2 + awgn_noise_std^2 ) ;
                d011 =  norm(r-H*s(:,4))^2/(H_noise_std^2*norm(s(:,4))^2 + awgn_noise_std^2) + color_num*log( H_noise_std^2*norm(s(:,4))^2 + awgn_noise_std^2 ) ;
                d100 =  norm(r-H*s(:,5))^2/(H_noise_std^2*norm(s(:,5))^2 + awgn_noise_std^2) + color_num*log( H_noise_std^2*norm(s(:,5))^2 + awgn_noise_std^2 ) ;
                d101 =  norm(r-H*s(:,6))^2/(H_noise_std^2*norm(s(:,6))^2 + awgn_noise_std^2) + color_num*log( H_noise_std^2*norm(s(:,6))^2 + awgn_noise_std^2 ) ;
                d110 =  norm(r-H*s(:,7))^2/(H_noise_std^2*norm(s(:,7))^2 + awgn_noise_std^2) + color_num*log( H_noise_std^2*norm(s(:,7))^2 + awgn_noise_std^2 ) ;
                d111 =  norm(r-H*s(:,8))^2/(H_noise_std^2*norm(s(:,8))^2 + awgn_noise_std^2) + color_num*log( H_noise_std^2*norm(s(:,8))^2 + awgn_noise_std^2 ) ;
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
save('RGB_same_h_constellation_design_vs_AE_multi_CSI_ML.mat')

