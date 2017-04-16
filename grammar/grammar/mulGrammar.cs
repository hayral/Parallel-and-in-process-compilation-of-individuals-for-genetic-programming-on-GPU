using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace grammar
{
    public class MulGrammar : Grammar
    {
        public override string bnf
        {
            get
            {
                return @"
<start>     ::= o0=<expr>;o1=<expr>;o2=<expr>;o3=<expr>;o4=<expr>;o5=<expr>;o6=<expr>;o7=<expr>;o8=<expr>;o9=<expr>;
<expr>      ::= (<expr2> <bi-op> <expr2>) | <var> | (~ <var>)
<expr2>      ::= (<expr2> <bi-op> <expr2>) | <var> | (~ <var>) | <var> | (~ <var>)
<var>       ::= a0 | a1 | a2 | a3 | a4 | b0 | b1 | b2 | b3 | b4
<bi-op>     ::= & | #or#
";
            }
        }

        public override string template
        {
            get
            {
                return @"
extern ""C"" {
    #codepreamble#
    #codeblock#
}} ";
            }
        }

        public override string codePreamble
        {
            get
            {

                return @"
__global__ void createdFunc0(int *_OUTPUT)
{
int tid = blockIdx.x *blockDim.x + threadIdx.x;;

int a0 = tid & 0x1;
int a1 = (tid & 0x2) >> 1;
int a2 = (tid & 0x4) >> 2;
int a3 = (tid & 0x8) >> 3;
int a4 = (tid & 0x10) >> 4;
int b0 = (tid & 0x20) >> 5;
int b1 = (tid & 0x40) >> 6;
int b2 = (tid & 0x80) >> 7;
int b3 = (tid & 0x100) >> 8;
int b4 = (tid & 0x200) >> 9;

int o0,o1,o2,o3,o4,o5,o6,o7,o8,o9;
";
            }
        }

        public override string beforeIndividualTemplate
        {
            get
            {
                return @"";
            }
        }

        public override string afterIndividualTemplate
        {
            get
            {    
                return @"
_OUTPUT[#individualId#*NUMTESTCASE+tid]= (o0 & 0x1) | ((o1 & 0x1) << 1) | ((o2 & 0x1) << 2) | ((o3 & 0x1) << 3) | ((o4 & 0x1) << 4) | ((o5 & 0x1) << 5) | ((o6 & 0x1) << 6) | ((o7 & 0x1) << 7) | ((o8 & 0x1) << 8) | ((o9 & 0x1) << 9);
";
            }
        }

        public MulGrammar(int testCaseCount = 1024) : base()
        {
            startToken = getToken("start");
            createTestCases(testCaseCount);
            defines.Add("NUMTESTCASE", testCaseCount.ToString());
            PopCount.InitializeBitcounts(10);
        }



        public List<int> testCases = new List<int>();

        public void createTestCases(int testCaseCount)
        {
            for (int ss = 0; ss < testCaseCount; ss++)
            {
                int a = ss & 0xF;
                int b = (ss & 0xF0) >> 4;                
                testCases.Add(a*b);         
            }

            _INPUT = testCases.ToArray();  ///   ** INITIALIZE HOST BUFFERS **
        }

        public override bool minimizeFitness
        {
            get { return true; }
        }

        /// ---------------------------------------------------------------------------------


        public int[] _INPUT;
        public int[] O;
        public ManagedCuda.CudaDeviceVariable<int> d_O;

        public override void SetKernelParameters(ManagedCuda.CudaKernel kernel)
        {
            //kernel.SetConstantVariable("_INPUT", _INPUT);
        }

        public override object[] KernelParameters()
        {
            return new object[]{d_O.DevicePointer};
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
                int[] individualsResults = O.Skip(ki * testCases.Count).Take(testCases.Count).ToArray();
                i.fitness = 0;
                for (int tci = 0; tci < testCases.Count; tci++)
                {
                    i.fitness += (PopCount.popCount[individualsResults[tci] ^ testCases[tci]]);
                }
                i.elite = false;
            } 
        }

        public override void InitializeHostBuffers()
        {
            
        }

        public override bool kernelPerIndividual
        {
            get { return false; }
        }
    }
}
