#define JIT



using ManagedCuda.BasicTypes;
using ManagedCuda.NVRTC;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Threading;
using ManagedCuda;

namespace CompileServer
{
    class Program
    {


        public static MemoryMappedFile mmf;
        public static EventWaitHandle evt;
        public static EventWaitHandle evt2;
        public static int ID;

        

        static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                System.Console.WriteLine("Please enter a numeric argument.");
            }
            else
            {
                System.Console.WriteLine(args[0]);
                
                ID = Convert.ToInt32(args[0]);
                evt = new EventWaitHandle(false, EventResetMode.AutoReset, "compileprocessevent" + args[0]);
                evt2 = new EventWaitHandle(false, EventResetMode.AutoReset, "compileprocessevent2" + args[0]);
 
                mmf = MemoryMappedFile.OpenExisting("compileprocess" + args[0]);
                var stream = mmf.CreateViewStream();
                

#if JIT
                var ctx = new CudaContext();
#endif         
                byte[] arrLengthBytes = new byte[4];
  
                System.Console.WriteLine("Compiler process started...");
                while (true)
                {
   
                    if (!evt.WaitOne(4000))   /// wait for start signal
                        break;
    

                    //Read from shared mem

                    stream.Position = 0;
                    stream.Read(arrLengthBytes, 0, 4);
                    int arrLength = BitConverter.ToInt32(arrLengthBytes, 0);

                    byte[] srcBuffer = new byte[arrLength];
                    stream.Read(srcBuffer, 0, arrLength);
                    string src = GetString(srcBuffer);

                    stream.Position = 0;

                    //  - 1 -  Ptx Compile
                    byte[] ptx = srcToPtx(src);
     

                    //  - 2 -  JIT Compile                        
#if JIT
                    var cl = new CudaLinker();
                    cl.AddData(ptx,CUJITInputType.PTX,null,null);  
                    ptx = cl.Complete();
      
#endif
                    // Write back to shared mem
                    stream.Write(BitConverter.GetBytes(ptx.Length), 0, 4);
                    stream.Write(ptx, 0, ptx.Length);
                    stream.Flush();

 
                    evt2.Set();   ///  signal the end of compilation
                }
            }
        }

        private static IntPtr[] optionPtr = new IntPtr[] { Marshal.StringToHGlobalAnsi("-arch=compute_30"), Marshal.StringToHGlobalAnsi("-w") };

        private static byte[] srcToPtx(string src)
        {
            var prg = new nvrtcProgram();
            var res = NVRTCNativeMethods.nvrtcCreateProgram(ref prg, src, null, 0, null, null);

 

            res = NVRTCNativeMethods.nvrtcCompileProgram(prg, 2, optionPtr);

            SizeT ptxSize = new SizeT();
            
            res = NVRTCNativeMethods.nvrtcGetPTXSize(prg, ref ptxSize);
            byte[] ptx = new byte[ptxSize];
            res = NVRTCNativeMethods.nvrtcGetPTX(prg, ptx);
            return ptx;
        }

        static byte[] GetBytes(string str)
        {
            byte[] bytes = new byte[str.Length * sizeof(char)];
            System.Buffer.BlockCopy(str.ToCharArray(), 0, bytes, 0, bytes.Length);
            return bytes;
        }

        static string GetString(byte[] bytes)
        {
            char[] chars = new char[bytes.Length / sizeof(char)];
            System.Buffer.BlockCopy(bytes, 0, chars, 0, bytes.Length);
            return new string(chars);
        }



 

    }

 
}
