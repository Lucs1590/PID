# T1
a = [16, 3, 2, 13;
     5, 10, 11, 8;
     9, 6, 7, 12;
     4, 15, 14, 1;]

b = [ 16 3 2 4
20 30 4 4
5 6 7 8]

c = [ 20,30 40;
50,40,10
80 20 15]

d = [ 1,2,3,4;5 6 7 8
9 9 9 9
8,8,8,8;
4 5,9 4]

# T2
whos
AI = uint8(a); BI = uint16(b); CI = uint32(c); DS = single(d);

# T3
a(2,3)
a(:,3)
a(1:2,1:3)
a(:,4) = 0
a(end,end)
a(end,end-2)
a(2:end,end:-2:1)
a(:)
a(:)'

# T4
# get elements of line 1 and 3, at colunm 2 and 3, that is, elements into colunm
# 2 and 3 that be on line 1 and 3 of matrix a
A6 = A([1 3], [2 3])
AL = logical([1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1])
a(AL)
a(3,3) == a(11)
# sum all elements
S2 = sum(sum(a))