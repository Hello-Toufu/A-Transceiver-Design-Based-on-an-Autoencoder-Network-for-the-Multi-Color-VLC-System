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
% [0.5620903  0.40524313 0.22477785 0.23294422]
% [0.5330704  0.4966505  0.26875135 0.25000623]
% [0.20141599 0.15418735 0.11767285 0.07836498]
H = InterferenceMatrix * diag([0.5330704  0.4966505  0.26875135 0.25000623]./[0.021,0.014,0.005,0.015]);

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
    ber(i_p_tx)=ber_sum/num_loop/random_symbol_num_per_color;
    %**********************************************************************
end
figure;
semilogy(SNR_dB, ber, 'b-o');
hold on;
grid on



