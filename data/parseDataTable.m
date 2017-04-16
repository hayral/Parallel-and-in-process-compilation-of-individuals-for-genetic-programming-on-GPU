function [ pop,val,err ] = parseDataTable( data , type)
err = [];
val=[];
pop=unique(data.PopulationSize)';
for p = pop
    other = data{(data{:,3} == type)&(data.PopulationSize == p),5:end}; %data{(data.Total1Ptx2Jit3 == 1)&(data.PopulationSize == pop),5:end};
    val(end+1) = mean2(other)/p;
    err(end+1) = std2(other)/p;
end

end

