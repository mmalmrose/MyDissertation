function my_rebin, inarray, nbins
thesize = n_elements(inarray)
inbin = thesize/nbins
output=0
for i = 0, nbins-2 do begin
starthere = i*inbin
endhere = (i+1)*inbin
thisvalue = mean(inarray(starthere:endhere))
output = [output, thisvalue]
endfor
output = output(1:*)
finalval = mean(inarray(endhere:*))
output = [output, finalval]
return, outp
end
