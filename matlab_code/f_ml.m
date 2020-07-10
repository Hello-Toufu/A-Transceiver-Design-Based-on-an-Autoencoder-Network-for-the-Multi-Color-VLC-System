function x_0=f_ml(r,x,H,sigma_h,sigma_n,vector_size)
% golden search
% intervals
a = 0;  
b = 100;  
Err = 0.0000000001;  
T = 0.618;  % golden factor
c = a+(1-T)*(b-a);  
d = b-(1-T)*(b-a);  
[Fc,x_0] = f_unimodal_optimization(c,r,x,H,sigma_h,sigma_n,vector_size);  
[Fd,x_0] = f_unimodal_optimization(d,r,x,H,sigma_h,sigma_n,vector_size);  
while(1)  
    if(abs(b-a)<Err)  
        y=0.5*(a+b);  
        break;  
    end  
    if(Fc<Fd)  
        a = a;  
        b = d;  
        d = c;  
        Fd = Fc;  
        %%%%%%%%%  
        c = a+(1-T)*(b-a);  
        [Fc,x_0] = f_unimodal_optimization(c,r,x,H,sigma_h,sigma_n,vector_size);  
    else %Fc>Fd  
        b = b;  
        a = c;  
        c = d;  
        Fc = Fd;  
        %%%%%%%%%%  
        d = b-(1-T)*(b-a);  
        [Fd,x_0] = f_unimodal_optimization(d,r,x,H,sigma_h,sigma_n,vector_size);  
    end      
end  

end