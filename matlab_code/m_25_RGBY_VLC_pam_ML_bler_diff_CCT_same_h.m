%===============================================================================
% # Author            :   ZDF
% # Created on        :
% # last modified     :   12/12/2019 Thu
% # Description       :
% # 3. RGBY
%   (finally used in paper) bler performance comparision : MIMO-PAM (ML) vs.
%   constellation design (ML) vs. AE
%===============================================================================
clear;clc;
%===============================================================================
% parameters
%===============================================================================
random_symbol_num_per_color=1e6;
color_num = 4;
SNR_dB     =   [0 : 2 : 20];
M = 2; % modulation order
xi = 0.1;
InterferenceMatrix = [(1-xi)    xi             0           0
    xi      (1-2*xi)      xi          0
    0       xi          (1-2*xi)    xi
    0       0            xi          (1-xi)];

%6500K [1.1408 ; 1.3226 ; 1]
%5000K [1.6339 ;   1.6348 ;   1.0000]
%4000 [2.3762  ;  2.0844  ;  1.0000]
%2700 [6.1892  ;  3.9912  ;  1.0000]
% 6500 5000 2700
ber_all=[];
for i=1
    %******************************************
    if i ==1
        H = InterferenceMatrix * diag([0.41513863 0.41522923 0.42026663 0.26378915]);
    elseif  i==2
        H = InterferenceMatrix * diag([0.42053717 0.4210558  0.37056506 0.1898223 ]);
    elseif  i==3
        H = InterferenceMatrix * diag([0.36526972 0.36935332 0.19994046 0.04334819]);
        %     elseif  i==4
        %         H = InterferenceMatrix * diag([0.8124868  0.8124848  0.43449393 0.25000235]);
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
                d0000 = norm( H*real(pammod([0;0;0;0],M))-r ) ;
                d0001 = norm( H*real(pammod([0;0;0;1],M))-r ) ;
                d0010 = norm( H*real(pammod([0;0;1;0],M))-r ) ;
                d0011 = norm( H*real(pammod([0;0;1;1],M))-r ) ;
                d0100 = norm( H*real(pammod([0;1;0;0],M))-r ) ;
                d0101 = norm( H*real(pammod([0;1;0;1],M))-r ) ;
                d0110 = norm( H*real(pammod([0;1;1;0],M))-r ) ;
                d0111 = norm( H*real(pammod([0;1;1;1],M))-r ) ;
                d1000 = norm( H*real(pammod([1;0;0;0],M))-r ) ;
                d1001 = norm( H*real(pammod([1;0;0;1],M))-r ) ;
                d1010 = norm( H*real(pammod([1;0;1;0],M))-r ) ;
                d1011 = norm( H*real(pammod([1;0;1;1],M))-r ) ;
                d1100 = norm( H*real(pammod([1;1;0;0],M))-r ) ;
                d1101 = norm( H*real(pammod([1;1;0;1],M))-r ) ;
                d1110 = norm( H*real(pammod([1;1;1;0],M))-r ) ;
                d1111 = norm( H*real(pammod([1;1;1;1],M))-r ) ;
                min_d=min([d0000 d0001 d0010 d0011 d0100 d0101 d0110 d0111 ...
                    d1000 d1001 d1010 d1011 d1100 d1101 d1110 d1111]')';
                if(min_d==d0000)
                    y=[0;0;0;0];
                elseif(min_d==d0001)
                    y=[0;0;0;1];
                elseif(min_d==d0010)
                    y=[0;0;1;0];
                elseif(min_d==d0011)
                    y=[0;0;1;1];
                elseif(min_d==d0100)
                    y=[0;1;0;0];
                elseif(min_d==d0101)
                    y=[0;1;0;1];
                elseif(min_d==d0110)
                    y=[0;1;1;0];
                elseif(min_d==d0111)
                    y=[0;1;1;1];
                elseif(min_d==d1000)
                    y=[1;0;0;0];
                elseif(min_d==d1001)
                    y=[1;0;0;1];
                elseif(min_d==d1010)
                    y=[1;0;1;0];
                elseif(min_d==d1011)
                    y=[1;0;1;1];
                elseif(min_d==d1100)
                    y=[1;1;0;0];
                elseif(min_d==d1101)
                    y=[1;1;0;1];
                elseif(min_d==d1110)
                    y=[1;1;1;0];
                else
                    y=[1;1;1;1];
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
%===============================================================================
% results visualization
%===============================================================================
figure;
% MIMO-PAM
ber_all(1,:) = [0.518926300000000,0.383074400000000,0.239945400000000,0.119593300000000,0.0446098000000000,0.0116943000000000,0.00189070000000000,0.000125500000000000,2.20000000000000e-06,0,0];

semilogy(SNR_dB, ber_all(1,:), 'b--o');
%[0.518926300000000,0.383074400000000,0.239945400000000,0.119593300000000,0.0446098000000000,0.0116943000000000,0.00189070000000000,0.000125500000000000,2.20000000000000e-06,0,0]
hold on;
semilogy(SNR_dB, ber_all(2,:), 'b--s');
semilogy(SNR_dB, ber_all(3,:), 'b--x');

% constellation design
ber_all_constellation_design = [0.511087000000000,0.371387000000000,0.223980000000000,0.100157000000000,0.0280870000000000,0.00392400000000000,0.000206000000000000,2.00000000000000e-06,0,0,0;0.536473000000000,0.401150000000000,0.253777000000000,0.123072000000000,0.0392130000000000,0.00662800000000000,0.000462000000000000,7.00000000000000e-06,0,0,0;0.628817000000000,0.518690000000000,0.386722000000000,0.243156000000000,0.116042000000000,0.0365990000000000,0.00626200000000000,0.000387000000000000,5.00000000000000e-06,0,0];
semilogy(SNR_dB, ber_all_constellation_design(1,:), 'k-.o');
semilogy(SNR_dB, ber_all_constellation_design(2,:), 'k-.s');
semilogy(SNR_dB, ber_all_constellation_design(3,:), 'k-.x');

% AE
semilogy(SNR_dB, [5.03088999e-01 3.58285004e-01 2.07407998e-01 8.55889998e-02...
    2.11870000e-02 2.45200002e-03 8.49999997e-05 9.99999975e-07...
    0.00000000e+00 0.00000000e+00 0.00000000e+00], 'r-o');
semilogy(SNR_dB, [5.26014000e-01 3.84658000e-01 2.32412000e-01 1.03293000e-01...
    2.85440000e-02 3.78400004e-03 1.65000003e-04 2.99999992e-06...
    0.00000000e+00 0.00000000e+00 0.00000000e+00], 'r-s');
semilogy(SNR_dB, [5.98992002e-01 4.80759004e-01 3.41741997e-01 2.01203999e-01...
    8.93239990e-02 2.67719999e-02 4.81199999e-03 4.28999992e-04...
    1.49999996e-05 9.99999975e-08 0.00000000e+00], 'r-x');
% title('RGBY')
ylim([0.000006,1]);
grid on
legend('baseline 1 (6500K)','baseline 1 (5000K)','baseline 1 (2700K)','baseline 2 (6500K)','baseline 2 (5000K)','baseline 2 (2700K)','proposed scheme (6500K)','proposed scheme (5000K)','proposed scheme (2700K)')
xlabel('SNR (dB)')
ylabel('BLER')
% save workspace
save('RGBY_same_h.mat')

