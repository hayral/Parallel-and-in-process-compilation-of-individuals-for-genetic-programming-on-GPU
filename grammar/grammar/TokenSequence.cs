using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Collections;

namespace grammar
{
    public class TokenSequence : List<Object>
    {
        public TokenSequence(): base()
        {
            
        }

        public TokenSequence(string singleLiteralToken) : base(1)
        {
            Add(singleLiteralToken);
        }

        public TokenSequence(Token singleToken) : base(1)
        {
            Add(singleToken);
        }

        public TokenSequence(List<Object> tokens)
            : base(tokens)
        {
            //this.AddRange(tokens);
        }


        public override string ToString()
        {
            System.Text.StringBuilder sb = new System.Text.StringBuilder();
            foreach (Object s in this)
            {
                sb.Append(s.ToString());
            }
            return sb.ToString();
        }


    }
}
