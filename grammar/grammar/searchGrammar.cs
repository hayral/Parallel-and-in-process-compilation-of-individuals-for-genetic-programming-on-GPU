using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace grammar
{
    public class SearchGrammar : Grammar
    {
        public override string bnf
        {
            get
            {
                return @"
<expr>              ::= <expr2> <bi-op> <expr2> | <expr2>
<expr2>             ::= <int> | <var-read> | <var-indexed>

<var-read>          ::= tmp | i | OUTPUT | SEARCH
<var-indexed>       ::= INPUT[<var-read> % LENINPUT]
<var-write>         ::= tmp | OUTPUT

<bi-op>             ::= + | -

<int>               ::= 1 | 2 | (-1)

<statement>         ::= <assignement> | <if> | <loop>
<statement2>         ::= <assignement> | <if2>
<statement3>         ::= <assignement>

<loop>              ::= for(i=0;i #lesser# LENINPUT;i++){<c-block2>}

<if>                ::= if (<cond-expr>) {<c-block2>}
<if2>                ::= if (<cond-expr>) {<c-block3>}

<cond-expr>         ::= <expr> <comp-op> <expr>

<comp-op>           ::= #lesser# | #greater# | == | !=

<assignement>       ::= <var-write> = <expr>;

<c-block>           ::= <statements>
<c-block2>           ::= <statements2>
<c-block3>           ::= <statements3>

<statements>        ::= <statement>#nl# | <statement>#nl#<statement>#nl# | <statement>#nl#<statement>#nl#<statement>#nl#
<statements2>        ::= <statement2>#nl# | <statement2>#nl#<statement2>#nl# | <statement2>#nl#<statement2>#nl#<statement2>#nl#
<statements3>        ::= <statement3>#nl# | <statement3>#nl#<statement3>#nl# | <statement3>#nl#<statement3>#nl#<statement3>#nl#
";
            }
        }


        public override string template{
            get
            {
                return @"
extern ""C"" {
    #codepreamble#
    #codeblock#
}} ";
            }
        }




        public override string codePreamble{
            get
            {
                return @"
__constant__ int _INPUT[NUMTESTCASE][MAX_TESTCASE_LEN];
__constant__ int _LENINPUT[NUMTESTCASE];
__constant__ int _SEARCH[NUMTESTCASE];
__constant__ int CORRECTANSWER[NUMTESTCASE];


__global__ void createdFunc0(int *_OUTPUT)
{
    int *INPUT = _INPUT[threadIdx.x];
    int LENINPUT = _LENINPUT[threadIdx.x];
    int SEARCH = _SEARCH[threadIdx.x];
    
    int i;
    int tmp;
    int OUTPUT;
";  
            }
        }




        public override string beforeIndividualTemplate{
            get
            {
                return @"

    i = 0;
    tmp = 0;
    OUTPUT = -1;
";
            }
        }


        public override string afterIndividualTemplate{
            get
            {
                return @"
    _OUTPUT[#individualId#*blockDim.x+threadIdx.x]=((OUTPUT==CORRECTANSWER[threadIdx.x])?1:0);
";  //
            }
        }




        public SearchGrammar(int testCaseCount = 10,int maxTestCaseLength = 20) : base()
        {
            startToken = getToken("c-block");
            createTestCases(testCaseCount,maxTestCaseLength);
            defines.Add("NUMTESTCASE", testCaseCount.ToString());
            defines.Add("MAX_TESTCASE_LEN", maxTestCaseLength.ToString());
        }



        public List<Tuple<int[], int, int>> testCases = new List<Tuple<int[], int, int>>();

        public void createTestCases(int testCaseCount,int maxTestCaseLength)
        {
            var rnd = new Random();
            for (int i = 0; i < testCaseCount/2; i++)
            {
                var l = Enumerable.Range(0, rnd.Next(3, maxTestCaseLength)).Select(n => rnd.Next(1, 50)).ToArray<int>();
                int s = rnd.Next(0, l.Length);
                testCases.Add(Tuple.Create<int[], int, int>(l, l[s], s));
            }
            
            for (int i = 0; i < testCaseCount/2; i++)
            {
                var l = Enumerable.Range(0, rnd.Next(3, maxTestCaseLength)).Select(n => rnd.Next(1, 50)).ToArray<int>();
                int s = rnd.Next(1, 50);
                testCases.Add(Tuple.Create<int[], int, int>(l, s, Array.IndexOf<int>(l, s)));
            }            
        }

        public override bool minimizeFitness
        {
            get { return false; }
        }


        /// ----------------------------------------------------------------------------------------------

        public List<int> __INPUT = new List<int>();                     //  flatten into this
        public int[] _INPUT;
        public int[] _LENINPUT;
        public int[] _SEARCH;
        public int[] CORRECTANSWER;
        public int[] O;
        public ManagedCuda.CudaDeviceVariable<int> d_O;

        public override void SetKernelParameters(ManagedCuda.CudaKernel kernel)
        {
            kernel.SetConstantVariable("_INPUT", _INPUT);
            kernel.SetConstantVariable("_LENINPUT", _LENINPUT);
            kernel.SetConstantVariable("_SEARCH", _SEARCH);
            kernel.SetConstantVariable("CORRECTANSWER", CORRECTANSWER);
        }

        public override object[] KernelParameters()
        {
            return new object[] { d_O.DevicePointer };
        }

        public override void InitializeGPUBuffers(int POPSIZE)
        {
            d_O = new ManagedCuda.CudaDeviceVariable<int>(POPSIZE * testCases.Count);
        }

        public override void ReadBackFromGPUBuffersAndComputeFitness(List<Individual> population)
        {
            /// Read Back
            O = d_O;

            /// Fitness            
            for (int ki = 0; ki < population.Count; ki++) {
                Individual i = population[ki];
                i.fitness = O.Skip(ki * testCases.Count).Take(testCases.Count).Sum();
                i.elite = false;
            }            
        }

        public override void InitializeHostBuffers()
        {         
            __INPUT.Clear();

            int MAX_TESTCASE_LEN = Int32.Parse(defines["MAX_TESTCASE_LEN"]);
            foreach (var tc in testCases)           //  flatten
            {
                __INPUT.AddRange(tc.Item1);
                __INPUT.AddRange(Enumerable.Repeat<int>(0, MAX_TESTCASE_LEN  - tc.Item1.Length));  //  pad with zeros
            }
            _INPUT = __INPUT.ToArray();
            _LENINPUT = testCases.Select(x => x.Item1.Length).ToArray<int>();
            _SEARCH = testCases.Select(x => x.Item2).ToArray<int>();
            CORRECTANSWER = testCases.Select(x => x.Item3).ToArray<int>();        
        }

        public override bool kernelPerIndividual
        {
            get { return false; }
        }
    }
}
