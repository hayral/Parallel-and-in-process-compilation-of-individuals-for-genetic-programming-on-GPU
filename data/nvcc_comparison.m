% nvcc comparison of three problems
% Figure 1
load alldata+nvcc

perindividual=false;

data = datasearchnvcc;
[p,v1,e1] = parseDataTable(data,1);
[p,v2,e2] = parseDataTable(data,2);
[p,v3,e3] = parseDataTable(data,3);

errorbar(p,v1+v2+v3,e1+e2+e3);
if perindividual
    errorbar(p,v1+v2+v3,e1+e2+e3);
else
    errorbar(p,((v1+v2+v3).*p)/1000,((e1+e2+e3).*p)/1000);
end


hold on


data = dataK6nvcc;
[p,v1,e1] = parseDataTable(data,1);
[p,v2,e2] = parseDataTable(data,2);
[p,v3,e3] = parseDataTable(data,3);
if perindividual
    errorbar(p,v1+v2+v3,e1+e2+e3);
else
    errorbar(p,((v1+v2+v3).*p)/1000,((e1+e2+e3).*p)/1000);
end


data = dataMULnvcc;
[p,v1,e1] = parseDataTable(data,1);
[p,v2,e2] = parseDataTable(data,2);
[p,v3,e3] = parseDataTable(data,3);
if perindividual
    errorbar(p,v1+v2+v3,e1+e2+e3);
else
    errorbar(p,((v1+v2+v3).*p)/1000,((e1+e2+e3).*p)/1000);
end


xlabel('population size (#individuals)');
if perindividual
    ylabel('time (ms)');
else
    ylabel('time (sec)');
end
legend('Search Problem','Keijzer-6 Regression','5-bit Multiplier')

T = get(gca,'tightinset');
set(gca,'position',[T(1) T(2) 1-T(1)-T(3) 1-T(2)-T(4)]); 
