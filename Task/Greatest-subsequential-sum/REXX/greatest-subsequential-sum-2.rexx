/*REXX program finds the  longest  greatest continuous subsequence  sum.*/
parse arg @;         w=words(@)        /*get arg list;  # words in list.*/
say 'words='w    "   list="@           /*show #words &  LIST to console.*/
sum=0;  L=0;  at=w+1                   /*default sum, length, starts at.*/
                                       /* [↓]  process the list of nums.*/
  do j=1  for w;     f=word(@,j)       /*select one number at a time.   */
      do k=j  to w;  _=k-j+1           /* [↓]  process a sub─list of #s.*/
      s=f;           do m=j+1  to k;   s=s+word(@,m);    end  /*m*/
      if (s==sum & _>L)  |  s>sum  then do;  sum=s;   at=j;    L=_;    end
      end   /*k*/                      /* [↑] chose longest greatest sum*/
  end       /*j*/

$=subword(@,at,L);    if $==''  then $="[NULL]"         /*Englishize it.*/
say;  say 'sum='sum/1  "   sequence="$ /*stick a fork in it, we're done.*/