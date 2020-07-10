function [y,x_0]=f_unimodal_optimization(t,r,x,H,sigma_h,sigma_n,N)

% golden search
% intervals
a = -min(eig(H'*H));  
b = 100;  
Err = 0.0000000001;  
T = 0.618;  % golden factor
c = a+(1-T)*(b-a);  
d = b-(1-T)*(b-a);  
Fc = norm(norm(pinv( H'*H + c*eye(size(H'*H)) ) * H' * r)^2 - t);  
Fd = norm(norm(pinv( H'*H + d*eye(size(H'*H)) ) * H' * r)^2 - t);  
while(1)  
    if(abs(b-a)<Err)  
        eta=0.5*(a+b);  
        break;  
    end  
    if(Fc<Fd)  
        a = a;  
        b = d;  
        d = c;  
        Fd = Fc;  
        %%%%%%%%%  
        c = a+(1-T)*(b-a);  
        Fc = norm(norm(pinv( H'*H + c*eye(size(H'*H)) ) * H' * r)^2 - t);  
    else %Fc>Fd  
        b = b;  
        a = c;  
        c = d;  
        Fc = Fd;  
        %%%%%%%%%%  
        d = b-(1-T)*(b-a);  
        Fd = norm(norm(pinv( H'*H + d*eye(size(H'*H)) ) * H' * r)^2 - t);  
    end      
end  


x_0 = pinv( H'*H + eta*eye(size(H'*H)) ) * H' * r;
y = norm(r-H*x_0)^2/(sigma_h^2*t + sigma_n^2) + N*log( sigma_h^2*t + sigma_n^2 );

end