clear;clc;
%===============================================================================
% parameters
%===============================================================================
random_symbol_num_per_color=1e5;
color_num = 4;
SNR_dB     =   [0 : 5 : 20];
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
for i=1:3
    %******************************************
    if i ==1
        H = InterferenceMatrix * diag([3.8249311 5.0941973 4.0926642 2.9148004]./[0.021,0.014,0.005,0.015]);
    elseif  i==2
        H = InterferenceMatrix * diag([5.2150135 5.027457  4.244484  2.3893018]./[0.021,0.014,0.005,0.015]);
    elseif  i==3
        H = InterferenceMatrix * diag([7.859489  6.9225945 4.2096205 0.8633509]./[0.021,0.014,0.005,0.015]);
    elseif  i==4
        H = InterferenceMatrix * diag([0.8124868  0.8124848  0.43449393 0.25000235]./[0.021,0.014,0.005,0.015]);
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
% figure;
semilogy(SNR_dB, ber_all(1,:), 'b-o');
hold on;
grid on
semilogy(SNR_dB, ber_all(2,:), 'b-*');
semilogy(SNR_dB, ber_all(3,:), 'b-+');
% semilogy(SNR_dB, ber_all(4,:), 'b--');

semilogy(SNR_dB, [6.23099983e-01 3.04500014e-01 3.88999991e-02 9.99999975e-05     0.], 'r-o');
semilogy(SNR_dB, [6.33700013e-01 3.17000002e-01 3.83000001e-02 1.99999995e-04       0], 'r-*');
semilogy(SNR_dB, [6.26699984e-01 3.18699986e-01 4.36999984e-02 9.99999975e-05    0.], 'r-+');
% semilogy(SNR_dB, [0.72759998 0.53280002 0.3035     0.1569     0.0791    ], 'r--');
title('RGBY')

% legend('baseline 6500K','baseline 5000K','baseline 4000K','baseline 2700K','AE 6500K','AE 5000K','AE 4000K','AE 2700K')
legend('baseline 6500K','baseline 5000K','baseline 2700K','AE 6500K','AE 5000K','AE 2700K')
save('RGBY.mat')

