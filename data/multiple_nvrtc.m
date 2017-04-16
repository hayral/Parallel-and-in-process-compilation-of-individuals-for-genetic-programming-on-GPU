load alldata
data1 = dataMULnvrtc; 
data2 = dataMULnvrtc2;
data4 = dataMULnvrtc4;
data6 = dataMULnvrtc6;
data8 = dataMULnvrtc8;
data1 = dataK6nvrtc; 
data2 = dataK6nvrtc2;
data4 = dataK6nvrtc4;
data6 = dataK6nvrtc6;
data8 = dataK6nvrtc8;
data1 = datasearchnvrtc; 
data2 = datasearchnvrtc2;
data4 = datasearchnvrtc4;
data6 = datasearchnvrtc6;
data8 = datasearchnvrtc8;

perindividual=false;

[p,v,e] = parseDataTable(data1,2);
[p2,v2,e2] = parseDataTable(data1,3);

hold on
if perindividual
    errorbar(p2,(v+v2),(e+e2));
else
    errorbar(p2,(v+v2).*p,(e+e2).*p);
end


[p,v,e] = parseDataTable(data2,2);
[p2,v2,e2] = parseDataTable(data2,3);

if perindividual
    errorbar(p2,(v+v2),(e+e2));
else
    errorbar(p2,(v+v2).*p,(e+e2).*p);
end


[p,v,e] = parseDataTable(data4,2);
[p2,v2,e2] = parseDataTable(data4,3);

if perindividual
    errorbar(p2,(v+v2),(e+e2));
else
    errorbar(p2,(v+v2).*p,(e+e2).*p);
end


[p,v,e] = parseDataTable(data6,2);
[p2,v2,e2] = parseDataTable(data6,3);

if perindividual
    errorbar(p2,(v+v2),(e+e2));
else
    errorbar(p2,(v+v2).*p,(e+e2).*p);
end


[p,v,e] = parseDataTable(data8,2);
[p2,v2,e2] = parseDataTable(data8,3);

if perindividual
    errorbar(p2,(v+v2),(e+e2));
else
    errorbar(p2,(v+v2).*p,(e+e2).*p);
end

xlabel('population size (#individuals)');
ylabel('time (ms)');
legend('1','2','4','6','8')

T = get(gca,'tightinset');
set(gca,'position',[T(1) T(2) 1-T(1)-T(3) 1-T(2)-T(4)]); 

