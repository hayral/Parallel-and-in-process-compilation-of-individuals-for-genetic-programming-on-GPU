using System;
using System.Collections;
using System.Collections.Generic;

namespace grammar
{

    public class Token
    {
        public string name;

        public Grammar Grammar;
        public List<TokenSequence> expansionRules; 

        
        public Token(string name,Grammar Grammar)
        {
            this.name = name;
            this.Grammar = Grammar;
        }


        public void addExpansionRule(string stringLiteral)
        {
            expansionRules.Add(new TokenSequence(stringLiteral));
        }

        public void addExpansionRule(Token singleToken)
        {
            expansionRules.Add(new TokenSequence(singleToken));
        }

        public void addExpansionRule(List<Object> tokens)
        {
            expansionRules.Add(new TokenSequence(tokens));
        }




        public override string ToString()
        {
            return "<" + name+">";
        }

    }






}
