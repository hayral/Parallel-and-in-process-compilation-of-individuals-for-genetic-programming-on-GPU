using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace grammar
{
    public class PopCount
    {
        public static int[] popCount; 

        public static void InitializeBitcounts16()
        {
            popCount = new int[0x10000];
            int position1 = -1;
            int position2 = -1;
            for (int i = 1; i < 0x10000; i++, position1++)
            {
                if (position1 == position2)
                {
                    position1 = 0;
                    position2 = i;
                }
                popCount[i] = popCount[position1] + 1;
            }
        }

        public static void InitializeBitcounts(int bitlength)
        {
            popCount = new int[0x1 << bitlength];
            int position1 = -1;
            int position2 = -1;
            for (int i = 1; i < (0x1 << bitlength); i++, position1++)
            {
                if (position1 == position2)
                {
                    position1 = 0;
                    position2 = i;
                }
                popCount[i] = popCount[position1] + 1;
            }
        }

        public static int PrecomputedBitcount16(int value)
        {
            return popCount[value & 0xffff] + popCount[(value >> 16) & 0xffff];
        }
    }
}
