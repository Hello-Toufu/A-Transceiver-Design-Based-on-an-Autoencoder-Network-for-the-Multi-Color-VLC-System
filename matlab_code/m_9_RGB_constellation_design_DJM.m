clc
clear all;
close all;
% 28=1+2+3+4+5+6+7
n=1.5;                   %????????
FOV=60/180*pi;           %???
g=n^2/sin(FOV)^2;        %???????
T=1;                     %?????
A_PD=1e-4;                %PD??????1????,
phi=70/180*pi;          %????
m=log(1/2)/(log(cos(phi)));%??????
angle_irr=60/180*pi;     %???
angle_inc=40/180*pi;     %???
D1=2;                     %??LEd?PD???????LED??
RES_PD=[0.42;0.33;0.24];        %PD???RGB(A/W)
%3333333*********??********************%
H=(m+1)*A_PD*RES_PD*(cos(angle_irr))^(m)*cos(angle_inc)*T*g/(2*pi*D1^2);
lamda=0.2;               %????
G=[1-lamda lamda 0;lamda 1-2*lamda lamda;0 lamda 1-lamda];
H1=G*diag(H);

%===============================================================================
% parameters
%===============================================================================
random_symbol_num_per_color=1e5;
color_num = 3;
SNR_dB     =   [0 : 5 : 20];
% M = 2; % modulation order
xi = 0.1;
InterferenceMatrix = [(1-xi)    xi             0
    xi      (1-2*xi)      xi
    0        xi          (1-xi)];
H = InterferenceMatrix;% * diag(1./[0.0114; 0.0052; 0.0427]);
%******************************************
x0=0.313;y0=0.337;           %6500kcenter point
g11=86e4;g12=-40e4;g22=45e4;%????
%x0=0.346;y0=0.359;           %5000kcenter point
%g11=56e4;g12=-25e4;g22=28e4;%????
%x0=0.459;y0=0.412;           %2700kcenter point
%g11=40e4;g12=-19.5e4;g22=28e4;%????
%x0=0.380;y0=0.380;           %4000kcenter point
%g11=39.5e4;g12=-21.5e4;g22=26e4;%????
ksi=7;                         %????????
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
x_old = [0.7006;0.1547;0.1440];
y_old=[0.2993;0.8059;0.0297];
% x_old = [0.69406;0.22965;0.12301];
% y_old=[0.30257;0.70992;0.09249];
a=x_old./y_old;
b=1./y_old;

%%%%%%%%%%%%%%%%%
Lt=100;    %????
M=8;      %????
J=[1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0;0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0;0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1]; %????
c=[0.0114;0.0052;0.0427];      %RGB???????????
I_M=diag(ones(M,1));           %M?????
I_max=[1;1;1];           %LED??????
%%
%%%%%??????????s0%%%%%%%%%%%%%%%%%
%??????
Xr=0.7006;Yr=0.2993;
Xg=0.1547;Yg=0.8059;
Xb=0.1440;Yb=0.0297;
A1=[Xr/Yr,Xg/Yg,Xb/Yb;1,1,1;(1-Xr-Yr)/Yr,(1-Xg-Yg)/Yg,(1-Xb-Yb)/Yb];
b_0=[x0/y0;1;(1-x0-y0)/y0];
% ??????????
ratio_0=(A1^-1)*b_0;       %??????????
L0=Lt*ratio_0;             %??????????
%%
Dmax=0;
for loop=1:1   %????100???
    %%
    %     while 1
    S=unifrnd(0,I_max(1),3,M);  %unifrnd?a,b,m,n??m*n?[a,b]????
    %         s1=M*L0.*c-sum(S,2);          %sum(S,2)?S?????
    %         if s1>=zeros(3,1)&s1<=I_max
    %             S=[S,s1];
    %             break;
    %         end
    %     end
    % S?????
    s0=reshape(S,3*M,1);   %?????
    %%
    l=combntns(1:M,2); %???
    row=size(l,1); %???
    E1=[];
    for i=1:row
        E(:,i)=I_M(:,l(i,1))-I_M(:,l(i,2));
        E1(:,:,i)=kron((E(:,i))',H);  %??????H2??diag(ones(3,1))
    end
    E1=sum(E1,1);
    %%
    %%*************CVX**********************%
    j=0;
    D_old = 0;
    while j<=10   %??????10?
        cvx_begin
        variables fai(3) s(3*M);
        variables t D;
        maximize D
        subject to
        for kk=1:28
            2*s0'* E1(:,:,kk)'* E1(:,:,kk)*s-s0'* E1(:,:,kk)'* E1(:,:,kk)*s0>=D^2;
        end
        
        m=1/alpha*((a'-x0*b')*cos(theta)+([1 1 1]-y0*b')*sin(theta))*fai;
        n=1/beta*(([1 1 1]-y0*b')*cos(theta)-(a'-x0*b')*sin(theta))*fai;
        0<=t<=ksi.*b'*fai;
        {[m;n],t} <In> lorentz(2);
        fai>=0;
        %         [1 1 1]*fai==Lt;
        1/M*J*(s.*kron(ones(M,1),1./c))==fai;
        0<=s<=I_max(1);
        cvx_end
        %         for kk=1:28
        %             aaaa(:,kk)=2*s0'* E1(:,:,kk)'* E1(:,:,kk)*s-s0'* E1(:,:,kk)'* E1(:,:,kk)*s0;
        %         end
        %         sqrt(min(aaaa))
        j=j+1;
        s0=s;
        if(norm((D - D_old)/sqrt(D^2+D_old^2),1) <= 0.001)
            break;
        end
        D_old = D;
    end
    %%
    %?100???????????
    if Dmax<D
        Dmax=D;
        s_opt=s;
        fai_opt=fai;
    end
end
s = reshape(s_opt,3,M);
current_mean = mean(s,2);
current_mean/current_mean(3)
DD = 100;
for i=1:row
    D_temp=norm(H*s(:,l(i,1))-H*s(:,l(i,2)));
    if D_temp<DD
        DD = D_temp;
    end
end
%===============================================================================
% ber caculation
%===============================================================================
ber_all=[];
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
        rec_data = H * modData;
        rec_data_with_noise = awgn(rec_data,SNR,'measured');
        %         rec_data_with_noise = rec_data;
        
        for t = 1:random_symbol_num_per_color
            r = rec_data_with_noise(:,t);
            d000 = norm( H*s(:,1)-r ) ;
            d001 = norm( H*s(:,2)-r ) ;
            d010 = norm( H*s(:,3)-r ) ;
            d011 = norm( H*s(:,4)-r ) ;
            d100 = norm( H*s(:,5)-r ) ;
            d101 = norm( H*s(:,6)-r ) ;
            d110 = norm( H*s(:,7)-r ) ;
            d111 = norm( H*s(:,8)-r ) ;
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
            ber_sum = ber_sum + (( y~=u_data(t))>0);
        end
    end
    % bit error num / loop num / group num
    ber(i_p_tx)=ber_sum/random_symbol_num_per_color/num_loop;
    %**********************************************************************
end
ber_all = [ber_all;ber];

semilogy(SNR_dB, ber_all(1,:), 'b-o');
hold on;
grid on


