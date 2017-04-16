using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace grammar
{
    public class K6Grammar : Grammar
    {
        public override string bnf
        {
            get
            {
                return @"
<e> ::= <e2> + <e2> | <e2> - <e2> | <e2> * <e2> | <e2> / <e2> | sqrtf(fabsf(<e2>)) | sinf(<e2>) | tanhf(<e2>) | expf(<e2>) | logf(fabsf(<e2>)+1) | x | x | x | x | <c><c>.<c><c> | <c><c>.<c><c> | <c><c>.<c><c> | <c><c>.<c><c>
<e2> ::= <e3> + <e3> | <e3> - <e3> | <e3> * <e3> | <e3> / <e3> | sqrtf(fabsf(<e3>)) | sinf(<e3>) | tanhf(<e3>) | expf(<e3>) | logf(fabsf(<e3>)+1) | x | x | x | x | x | x | <c><c>.<c><c> | <c><c>.<c><c> | <c><c>.<c><c> | <c><c>.<c><c> | <c><c>.<c><c> | <c><c>.<c><c>
<e3> ::= <e3> + <e3> | <e3> - <e3> | <e3> * <e3> | <e3> / <e3> | sqrtf(fabsf(<e3>)) | sinf(<e3>) | tanhf(<e3>) | expf(<e3>) | logf(fabsf(<e3>)+1) | x | x | x | x | x | x | x | x | <c><c>.<c><c> | <c><c>.<c><c> | <c><c>.<c><c> | <c><c>.<c><c> | <c><c>.<c><c> | <c><c>.<c><c> | <c><c>.<c><c> | <c><c>.<c><c>
<c> ::= 0|1|2|3|4|5|6|7|8|9
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

__global__ void createdFunc0(float *_OUTPUT)
{
int x = blockIdx.x *blockDim.x + threadIdx.x;

";
            }
        }

        public override string beforeIndividualTemplate
        {
            get
            {
                return @"_OUTPUT[#individualId#*NUMTESTCASE+x] =";
            }
        }

        public override string afterIndividualTemplate
        {
            get
            {
                return @";
";
            }
        }

        public K6Grammar(int testCaseCount = 50) : base()
        {
            startToken = getToken("e");
            createTestCases(testCaseCount);
            defines.Add("NUMTESTCASE", testCaseCount.ToString());
        }



        public List<float> testCases = new List<float>();

        public void createTestCases(int testCaseCount)
        {
            float t = 0;
            for (int x = 1; x < testCaseCount+1; x++)
            {
                t += 1F/x;
                testCases.Add( t );
            }
        }

        public override bool minimizeFitness
        {
            get { return true; }
        }

        /// ---------------------------------------------------------------------------------


        public float[] O;
        public ManagedCuda.CudaDeviceVariable<float> d_O;

        public override void SetKernelParameters(ManagedCuda.CudaKernel kernel)
        {
            // pass
        }

        public override object[] KernelParameters()
        {
            return new object[]{d_O.DevicePointer};
        }

        public override void InitializeGPUBuffers(int POPSIZE)
        {
            d_O = new ManagedCuda.CudaDeviceVariable<float>(POPSIZE * testCases.Count);
        }



        public override void ReadBackFromGPUBuffersAndComputeFitness(List<Individual> population)
        {
            /// Read Back
            O = d_O;

            /// Fitness
            for (int ki = 0; ki < population.Count; ki++)
            {
                Individual i = population[ki];
                double fitness = 0;
                float[] answers = O.Skip(ki * testCases.Count).Take(testCases.Count).ToArray();   ///   result segment for individual
                for (int tci = 0; tci < testCases.Count; tci++)   ///   RMSE
                {
                    fitness += Math.Pow(answers[tci] - testCases[tci],2);
                }
                i.fitness = Math.Sqrt( fitness/testCases.Count);
                i.elite = false;
            } 
        }

        public override void InitializeHostBuffers()
        {
            //pass
        }

        public override bool kernelPerIndividual
        {
            get { return false; } /// if false it means all individuals are in one kernel
        }
    }
}
