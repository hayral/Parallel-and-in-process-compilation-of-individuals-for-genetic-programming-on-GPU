using System.Collections.Generic;
using System;
using System.Collections;
using System.Linq;


namespace grammar
{
    // ----- Grammar
    public abstract class Grammar
    {
        public Token startToken; // root symbol is expected to be assigned on subclasses
        public Dictionary<string, Token> tokenDictionary = new Dictionary<string,Token>();


        
        public abstract string bnf {get;}
        public abstract bool minimizeFitness { get; }

        public abstract string template { get; }
        public abstract string codePreamble { get; }
        public abstract string beforeIndividualTemplate { get; }
        public abstract string afterIndividualTemplate { get; }

        public abstract bool kernelPerIndividual { get; }

        public abstract void SetKernelParameters(ManagedCuda.CudaKernel kernel);
        public abstract object[] KernelParameters();
        public abstract void InitializeGPUBuffers(int POPSIZE);
        public abstract void InitializeHostBuffers();
        public abstract void ReadBackFromGPUBuffersAndComputeFitness(List<Individual> population);
        
        public Dictionary<string, string> defines = new Dictionary<string,string>();
        

        public string tokenToCache = "assignement";

        public Grammar()
        {
            if (bnf == "")
            {
                throw new Exception("Grammar subclasses must populate the bnf field!");
            }
            else parseBNF(bnf);
        }



        

        public Token createToken(string name)
        {
            Token t = new Token(name,this);
            tokenDictionary.Add(name, t);
            return t;
        }

        public Token getToken(string name)
        {
            return tokenDictionary[name];
        }

        public Token getOrCreateToken(string name)
        {
            if (tokenDictionary.ContainsKey(name))
            {
                return tokenDictionary[name];
            }
            else return createToken(name);
        }

        

        // backwards scanning in loop here provides Breadth First Traversal with FIFO ordered expansions
        // this is a form of recursion removal
        public string compileGenotypeToCodeV1(Individual i)  //  !!  expansion may go to infinity
        {
            var program = new LinkedList<Object>();
            program.AddFirst(startToken);
            while (program.Any(o => (o is Token))) {  
                var node = program.Last;  //  start from last
                while(node != null){
                    if (node.Value is Token)
                    {
                        Token t = (Token)node.Value;
                        var chosenRule = t.expansionRules[i.readValueFromGenotype(t.expansionRules.Count)];
                        for(int x=chosenRule.Count-1;x>=0;x--){
                            program.AddAfter(node, chosenRule[x]);
                        }
                        var prevnode = node.Previous;
                        program.Remove(node);
                        node = prevnode;  //  keep scanning backwards
                    }
                    else { 
                        node = node.Previous;  //  scan backwards
                    }
                }
            }

            return String.Join("", program);
        }
        
        /// ------------------------------------------------  RECURSIVE EXPANSION  ----------------------------------------------------------------------
        
        public TokenSequence expandRecursive(Token t, Individual i){
            TokenSequence codeFragment = new TokenSequence( t.expansionRules[i.readValueFromGenotype(t.expansionRules.Count)]); // shallow copy
            for(int x=0;x<codeFragment.Count;x++)
            {
                if (codeFragment[x] is Token) codeFragment[x] = expandRecursive((Token)codeFragment[x], i);
            }

            return codeFragment;            
        }

        public string compileGenotypeToCode(Individual i)  //  !!  expansion may go to infinity
        {
            string s = String.Join("", expandRecursive(startToken, i));
            return s;
        }

        /// ----------------------------------------------------------------------------------------------------------------------

        public List<String> generateAllExpansions(Object token)
        {
            var result = new List<String>();
            if (token is Token) {
                foreach (TokenSequence expansionRule in ((Token)token).expansionRules)
                {
                    if (expansionRule.Count == 1)
                    {
                        if (expansionRule[0] is Token) 
                            result.AddRange(generateAllExpansions((Token)expansionRule[0]));
                        else 
                            result.Add(expansionRule[0].ToString());
                    }
                    else
                    {
                        var partialCartesianProduct = generateAllExpansions(expansionRule[0]);
                        for (int i = 1; i < expansionRule.Count; i++)
                        {
                            var expansionsB = generateAllExpansions(expansionRule[i]);
                            var newPartialCartesianProduct = new List<String>();
                            foreach (var A in partialCartesianProduct)
                            {
                                foreach (var B in expansionsB)
                                {
                                    newPartialCartesianProduct.Add(A + B);
                                }
                            }
                            partialCartesianProduct = newPartialCartesianProduct;
                        }
                        result.AddRange(partialCartesianProduct);
                    }
                }
            } else 
                result.Add(token.ToString());
            return result;
        }
        
        /// ----------------------------------------------------------------------------------------------------------------------


        public void parseBNF(string bnf)
        {
            var lines = bnf.Split(new string[] { "\r\n","\n" }, StringSplitOptions.RemoveEmptyEntries);
            var delimiters = "<>".ToCharArray();            

            var rules = new Dictionary<string, string[]>();
            foreach (string line in lines)
            {
                var splitRule = line.Split(new string[] { "::=" }, StringSplitOptions.None);
                var nonterminalName = splitRule[0].Trim().Trim(delimiters);
                rules.Add(nonterminalName, splitRule[1].Trim().Split("|".ToCharArray(), StringSplitOptions.RemoveEmptyEntries));                
                createToken(nonterminalName);
            }

            var parsedRules = new Dictionary<string, List<TokenSequence>>();            
            foreach (var kv in rules)
            {
                getToken(kv.Key).expansionRules = parseRule(kv.Value);  //  expansion rules in tokens
            }            
        }

        public List<TokenSequence> parseRule(string[] rule)
        {
            var tokenSequences = new List<TokenSequence>();
            TokenSequence ts;
            int a,b;
            foreach (string ss in rule)  //  s | s | s |
            {                
                if (ss.Contains("<")) {
                    string s = ss;
                    ts = new TokenSequence();
                    while (s.Contains("<"))
                    {
                        a = s.IndexOf("<");
                        b = s.IndexOf(">");
                        string leftstring = s.Substring(0, a);
                        if(leftstring.Trim().Length > 0) ts.Add(leftstring);  //  ....<                        
                        ts.Add(getToken(s.Substring(a + 1, b - a - 1)));  //  ....<....>
                        s = s.Remove(0, b + 1);
                    }
                    if (s.Trim().Length > 0) ts.Add(s);

                } else {
                    ts = new TokenSequence(ss);
                }
                tokenSequences.Add(ts);                
            }
            return tokenSequences;
        }

        public override string ToString()
        {
            System.Text.StringBuilder sb = new System.Text.StringBuilder();
            foreach(Token t in tokenDictionary.Values) {
                sb.Append(t.ToString()).Append(" ::= ");
                foreach(var er in t.expansionRules) {
                    sb.Append(er.ToString()).Append(" | ");
                }
                sb.Remove(sb.Length-3,3);
                sb.AppendLine();
            }
            return sb.ToString();
        }


    }


}
