load alldata


% choose one from below
data11 = datasearchnvcc; 
data12 = datasearchnvrtc;

data21 = dataK6nvcc; 
data22 = dataK6nvrtc; 

data31 = dataMULnvcc; 
data32 = dataMULnvrtc; 

perindividual=false;

hold on

[p,v,e] = parseDataTable(data11,2);
[p2,v2,e2] = parseDataTable(data11,3);
if perindividual
    errorbar(p2,(v+v2),(e+e2));
else
    errorbar(p2,.001*(v+v2).*p,.001*(e+e2).*p);
end

[p,v,e] = parseDataTable(data21,2);
[p2,v2,e2] = parseDataTable(data21,3);
if perindividual
    errorbar(p2,(v+v2),(e+e2));
else
    errorbar(p2,.001*(v+v2).*p,.001*(e+e2).*p);
end

[p,v,e] = parseDataTable(data21,2);
[p2,v2,e2] = parseDataTable(data21,3);
if perindividual
    errorbar(p2,(v+v2),(e+e2));
else
    errorbar(p2,.001*(v+v2).*p,.001*(e+e2).*p);
end


[p,v,e] = parseDataTable(data12,2);
[p2,v2,e2] = parseDataTable(data12,3);
if perindividual
    errorbar(p2,(v+v2),(e+e2));
else
    errorbar(p2,.001*(v+v2).*p,.001*(e+e2).*p);
end

[p,v,e] = parseDataTable(data22,2);
[p2,v2,e2] = parseDataTable(data22,3);
if perindividual
    errorbar(p2,(v+v2),(e+e2));
else
    errorbar(p2,.001*(v+v2).*p,.001*(e+e2).*p);
end

[p,v,e] = parseDataTable(data32,2);
[p2,v2,e2] = parseDataTable(data32,3);
if perindividual
    errorbar(p2,(v+v2),(e+e2));
else
    errorbar(p2,.001*(v+v2).*p,.001*(e+e2).*p);
end



xlabel('population size (#individuals)');
if perindividual
    ylabel('compile time per individual (ms)');
else
    ylabel('population (all individuals) compile time (sec)');
end
legend('out of process compilation','inprocess compilation')

T = get(gca,'tightinset');
set(gca,'position',[T(1) T(2) 1-T(1)-T(3) 1-T(2)-T(4)]); 