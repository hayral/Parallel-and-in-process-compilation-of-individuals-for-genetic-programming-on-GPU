using Cudafy;
using Cudafy.Compilers;
using grammar;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.NVRTC;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace managedcuda_try
{
    public class ExperimentV2
    {
        public int POPSIZE = 128;
        public int GENCOUNT = 100;
        public int NUMTESTCASE = 32;

        public int PARALLELISM_LEVEL = 2;
        public int GENOME_SIZE = 13 * 17;

        public double XORATE = 0.7;
        public double MUTATION_RATE = 0.7;



        public Grammar grm;                                         
        public GP gp;
        
        


        public List<SourceCode> sourceCodes = new List<SourceCode>();   

        
        public CudaContext ctx;                                        

        public List<CudaKernel> kernels = new List<CudaKernel>();
        public List<CudaStream> streams = new List<CudaStream>();




        public void RunExperiment(Grammar grm,bool nvrtc = true, bool remote = false)
        {

            Initialize(grm);  ///                                      ---  (1)  gp, grammar, population, individual


            // Evolution of generations
            for (int gen = 0; gen < GENCOUNT; gen++)
            {
                CreateSourceCodes(PARALLELISM_LEVEL);        ///    ---  (2)  create source
                
                PTXcompile(nvrtc, remote);  ///                     ---  (3) ptx compile
                 
                JITcompile();               ///                     ---  (4) JIT compile

                tic();
                
                CreateKernelObjects();      ///                     ---
                 
                LaunchKernels();            ///                     ---  (5)  launch kernel
                 
                WaitCompleteAndReadBack();  ///                     ---

                               

                GPoperations();             ///                     ---  (6) gp operations: selection, XO, mutation
                
                SampleCollector.Collect("other", toc());
                 
                
            }
            ctx.Dispose();
        }



        public void WaitCompleteAndReadBack()
        {
            ctx.Synchronize();  // wait all to finish

            grm.ReadBackFromGPUBuffersAndComputeFitness(gp.population);  //  read back from GPU memory


            Individual e = gp.getElite();
            e.elite = true;             // mark the new elite
            SampleCollector.Collect("fitness", e.fitness);
        }



        public void PTXcompile(bool nvrtc, bool remote)
        {
            tic();
            if (nvrtc)
            {
                if (remote)
                {
                    // Multiple remote NVRTC
                    PtxCompileRemoteMultiprocess();
                }
                else
                {
                    // Single thread, NVRTC                    
                    srcToPtx(sourceCodes[0]);  
                }
            }
            else
            {
                // Multiple NVCC launch
                PtxCompileNVCCMultiprocess();  
            }
            SampleCollector.Collect("PTX", toc());
        }



        public void JITcompile()
        {
            tic();
            foreach (var sc in sourceCodes)
            {
                
                if (sc.ptx == null)
                    sc.mod = ctx.LoadModulePTX(sc.cubin, null, null);  // no jit, just upload
                else
                    sc.mod = ctx.LoadModulePTX(sc.ptx); //jit 
            }

            SampleCollector.Collect("JIT", toc());
        }

        public void GPoperations()
        {
            gp.population = gp.tournamentSelection2(true);

            gp.crossover(XORATE);
            
            gp.mutate(MUTATION_RATE);
        }



        public void LaunchKernels()
        {
             int streamNo = 0;
             foreach (var kernel in kernels)
            {
                 kernel.RunAsync(streams[streamNo++].Stream, grm.KernelParameters());  /// (?)  mod streamcount to reuse streams if population size increase dynamically
             }
         }



        public void CreateKernelObjects()
        {
            
            kernels.Clear();
            int funcNo = 0;
            foreach (var sc in sourceCodes)
            {

                for (int ki = 0; ki < (grm.kernelPerIndividual? sc.numberOfIndividuals:1); ki++)
                {
                    CudaKernel kernel = new CudaKernel("createdFunc" + (grm.kernelPerIndividual ? funcNo++ : 0).ToString(), sc.mod, ctx); 
                     
                    if (NUMTESTCASE > 256)
                    {   // ---  multi dim block if testcases > 256
                        kernel.GridDimensions = NUMTESTCASE / 256;
                        kernel.BlockDimensions = 256;
                    }
                    else
                    {
                        kernel.GridDimensions = 1;
                        kernel.BlockDimensions = NUMTESTCASE;
                    }

                    if (ki == 0)  //   <--   this is due to managedcuda not implementing setconstantvar as a module method!
                        grm.SetKernelParameters(kernel);

                    kernels.Add(kernel);
                }
            }
        }



        /*
         * This function partitions individuals to SourceCode instances considering the number of former might not be divisible by the latter.
         */
        public void CreateSourceCodes(int parallelism_level = 1)
        {
            sourceCodes.Clear();
            int[] partitionSizes = new int[parallelism_level];
            for (int i = 0; i < parallelism_level; i++)
                partitionSizes[i] = POPSIZE / parallelism_level;
            int extra = POPSIZE - ((POPSIZE / parallelism_level) * parallelism_level);
            for (int i = 0; i < extra; i++)
                partitionSizes[i]++;


            int individualID = 0;
            for (int i = 0; i < parallelism_level; i++)
            {
                SourceCode sc = new SourceCode(i);
                sourceCodes.Add(sc);

                // append grammar defines to sourcecode defines
                foreach (var kv in grm.defines)
                    sc.defines.Add(kv.Key, kv.Value);

                sc.render(gp.population.Skip(individualID).Take(partitionSizes[i]), grm, individualID);
                individualID += partitionSizes[i];
            }
        }



        public void Initialize(Grammar grm)
        {
            InitializeGPandGrammar(grm);
            grm.InitializeHostBuffers();
            sourceCodes.Clear();
            InitializeGPU();
        }


        public void InitializeGPU()
        {
            ctx = new CudaContext();   
            kernels.Clear();
            streams.Clear();
            for (int ki = 0; ki < POPSIZE; ki++)           
                streams.Add(new CudaStream(CUStreamFlags.NonBlocking));

            grm.InitializeGPUBuffers(POPSIZE);
        }



        public void InitializeGPandGrammar(Grammar grm)
        {
            this.grm = grm;
            gp = new GP();
            gp.createPopulation(POPSIZE, GENOME_SIZE);
            gp.minimizeFitness = grm.minimizeFitness;
        }



        /// ==========================================================================================================



        public static Random rng = new Random();
        public void PtxCompileNVCCMultiprocess()  ///  NVCC
        {
            Parallel.ForEach(sourceCodes, (sc) =>
            {
                var km = new CudafyModule();
                km.SourceCode = sc.rendered;
                var co = NvccCompilerOptions.Createx64(new Version(8, 0), eArchitecture.sm_30);
                co.CompileMode = eCudafyCompileMode.Binary;
                co.AddOption("-w");
                km.CompilerOptionsList.Add(co);
                km.Compile(eGPUCompiler.CudaNvcc, false, eCudafyCompileMode.Binary, rng.Next().ToString());
                sc.cubin = km.Binary.Binary;
            });
        }

        // ---

        public void PtxCompileRemoteMultiprocess()  ///   ------------------------------   NVCC+multiple remote
        {
            foreach (var sc in sourceCodes)
                sc.StartCompile();

            // --- WaitAll workaround
            foreach (var sc in sourceCodes)
            {
                sc.Wait();
                sc.GatherCompileResult();
            }
        }

        // ---

        private static IntPtr[] optionPtr = new IntPtr[] { Marshal.StringToHGlobalAnsi("-arch=compute_30"), Marshal.StringToHGlobalAnsi("-w") }; 
        public static void srcToPtx(SourceCode sc)
        {
            var prg = new nvrtcProgram();
            var res = NVRTCNativeMethods.nvrtcCreateProgram(ref prg, sc.rendered, null, 0, null, null);

            res = NVRTCNativeMethods.nvrtcCompileProgram(prg, 2, optionPtr);
            SizeT ptxSize = new SizeT();
            res = NVRTCNativeMethods.nvrtcGetPTXSize(prg, ref ptxSize);
            sc.ptx = new byte[ptxSize];
            res = NVRTCNativeMethods.nvrtcGetPTX(prg, sc.ptx);
        }


        ///  -------------------------------------------------------------------------    auxilary code

        public static int _tictoc = 0;
        public static Stopwatch sw = new Stopwatch();

        public static void tic()
        {
            sw.Restart();
        }

        public static double toc()
        {
            return sw.Elapsed.TotalMilliseconds;
        }

        


    }
}
