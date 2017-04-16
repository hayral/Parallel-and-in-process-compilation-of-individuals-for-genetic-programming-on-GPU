﻿//	Copyright (c) 2014, Michael Kunz. All rights reserved.
//	http://kunzmi.github.io/managedCuda
//
//	This file is part of ManagedCuda.
//
//	ManagedCuda is free software: you can redistribute it and/or modify
//	it under the terms of the GNU Lesser General Public License as 
//	published by the Free Software Foundation, either version 2.1 of the 
//	License, or (at your option) any later version.
//
//	ManagedCuda is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//	GNU Lesser General Public License for more details.
//
//	You should have received a copy of the GNU Lesser General Public
//	License along with this library; if not, write to the Free Software
//	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
//	MA 02110-1301  USA, http://www.gnu.org/licenses/.


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace ManagedCuda
{
	/// <summary>
	/// Cuda occupancy from CudaOccupancy.h
	/// </summary>
	public class CudaOccupancy
	{
		const int MIN_SHARED_MEM_PER_SM = 16384;
		const int MIN_SHARED_MEM_PER_SM_GK210 = 81920;

		/// <summary>
		/// mirror the type and spelling of cudaDeviceProp's members keep these alphabetized
		/// </summary>
		public class cudaOccDeviceProp {
			/// <summary/>
			public int computeMajor;
			/// <summary/>
			public int computeMinor;
			/// <summary/>
			public int maxThreadsPerBlock;
			/// <summary/>
			public int maxThreadsPerMultiProcessor;
			/// <summary/>
			public int regsPerBlock;
			/// <summary/>
			public int regsPerMultiprocessor;
			/// <summary/>
			public int warpSize;
			/// <summary/>
			public SizeT sharedMemPerBlock;
			/// <summary/>
			public SizeT sharedMemPerMultiprocessor;
			/// <summary/>
			public int numSms; 

			/// <summary/>
			public cudaOccDeviceProp()
			{

			}

			/// <summary/>
			public cudaOccDeviceProp(int deviceID)
				: this(CudaContext.GetDeviceInfo(deviceID))
			{

			}

			/// <summary/>
			public cudaOccDeviceProp(CudaDeviceProperties props)
			{
				computeMajor = props.ComputeCapabilityMajor;
				computeMinor = props.ComputeCapabilityMinor;
				maxThreadsPerBlock = props.MaxThreadsPerBlock;
				maxThreadsPerMultiProcessor = props.MaxThreadsPerMultiProcessor;
				regsPerBlock = props.RegistersPerBlock;
				regsPerMultiprocessor = props.MaxRegistersPerMultiprocessor;
				warpSize = props.WarpSize;
				sharedMemPerBlock = props.SharedMemoryPerBlock;
				sharedMemPerMultiprocessor = props.MaxSharedMemoryPerMultiprocessor;
				numSms = props.MultiProcessorCount;
			}
		}

		
		/// <summary>
		/// define our own cudaOccFuncAttributes to stay consistent with the original header file
		/// </summary>
		public class cudaOccFuncAttributes
		{
			/// <summary/>
			public int maxThreadsPerBlock;
			/// <summary/>
			public int numRegs;
			/// <summary/>
			public SizeT sharedSizeBytes;
			/// <summary/>
			public cudaOccPartitionedGCConfig partitionedGCConfig;

			/// <summary>
			/// 
			/// </summary>
			public cudaOccFuncAttributes()
			{ 
			
			}

			/// <summary>
			/// cudaOccFuncAttributes
			/// </summary>
			/// <param name="aMaxThreadsPerBlock"></param>
			/// <param name="aNumRegs"></param>
			/// <param name="aSharedSizeBytes">Only the static part shared memory (without dynamic allocations)</param>
			/// <param name="partitionedGC"></param>
			public cudaOccFuncAttributes(int aMaxThreadsPerBlock, int aNumRegs, SizeT aSharedSizeBytes, cudaOccPartitionedGCConfig partitionedGC)
			{
				maxThreadsPerBlock = aMaxThreadsPerBlock;
				numRegs = aNumRegs;
				sharedSizeBytes = aSharedSizeBytes;
				partitionedGCConfig = partitionedGC;
			}

			/// <summary>
			/// 
			/// </summary>
			/// <param name="aKernel"></param>
			public cudaOccFuncAttributes(CudaKernel aKernel)
				: this(aKernel.MaxThreadsPerBlock, aKernel.Registers, aKernel.SharedMemory, cudaOccPartitionedGCConfig.Off)
			{ 
			
			}
		}

		/// <summary>
		/// Occupancy Error types
		/// </summary>
		public enum cudaOccError
		{
			/// <summary/>
			None = 0,
			/// <summary>
			/// input parameter is invalid
			/// </summary>
			ErrorInvalidInput = -1,
			/// <summary>
			/// requested device is not supported in current implementation or device is invalid
			/// </summary>
			ErrorUnknownDevice  = -2, 
		}

		
		/// <summary>
		/// Function cache configurations
		/// </summary>
		public enum cudaOccCacheConfig
		{
			/// <summary>
			/// no preference for shared memory or L1 (default) 
			/// </summary>
			PreferNone    = 0x00, 
			/// <summary>
			/// prefer larger shared memory and smaller L1 cache
			/// </summary>
			PreferShared  = 0x01, 
			/// <summary>
			/// prefer larger L1 cache and smaller shared memory
			/// </summary>
			PreferL1      = 0x02,
			/// <summary>
			/// prefer equal sized L1 cache and shared memory
			/// </summary>
			PreferEqual   = 0x03,
		}


		/// <summary>
		/// Occupancy Limiting Factors 
		/// </summary>
		[Flags]
		public enum cudaOccLimitingFactors
		{
			/// <summary>
			/// occupancy limited due to warps available
			/// </summary>
			Warps = 0x01, 
			/// <summary>
			/// occupancy limited due to registers available
			/// </summary>
			Registers = 0x02, 
			/// <summary>
			/// occupancy limited due to shared memory available
			/// </summary>
			SharedMemory = 0x04, 
			/// <summary>
			/// occupancy limited due to blocks available
			/// </summary>
			Blocks         = 0x08 
		};


		/// <summary>
		/// Partitioned global caching support
		/// </summary>
		public enum cudaOccPartitionedGCSupport
		{
			/// <summary>
			/// Partitioned global caching is not supported
			/// </summary>
			NotSupported, 
			/// <summary>
			/// Partitioned global caching is supported
			/// </summary>
			Supported, 
			/// <summary>
			/// This is only needed for Pascal. This, and
            /// all references / explanations for this,
            /// should be removed from the header before
            /// exporting to toolkit.
			/// </summary>
			AlwaysOn
		};


		/// <summary>
		/// Partitioned global caching option
		/// </summary>
		public enum cudaOccPartitionedGCConfig
		{
			/// <summary>
			/// Disable partitioned global caching
			/// </summary>
			Off, 
			/// <summary>
			/// Prefer partitioned global caching
			/// </summary>
			On, 
			/// <summary>
			/// Force partitioned global caching
			/// </summary>
			OnStrict
		};

		/// <summary>
		/// 
		/// </summary>
		public class cudaOccResult
		{
			/// <summary>
			/// Active Thread Blocks per Multiprocessor
			/// </summary>
			public int ActiveBlocksPerMultiProcessor;
			/// <summary/>
			public cudaOccLimitingFactors LimitingFactors;
			/// <summary/>
			public int BlockLimitRegs;
			/// <summary/>
			public int BlockLimitSharedMem;
			/// <summary/>
			public int BlockLimitWarps;
			/// <summary/>
			public int BlockLimitBlocks;
			/// <summary/>
			public int AllocatedRegistersPerBlock;
			/// <summary/>
			public int AllocatedSharedMemPerBlock;
			/// <summary/>
			public cudaOccPartitionedGCConfig partitionedGCConfig;
		}

		/// <summary>
		/// define cudaOccDeviceState to include any device property needed to be passed
		/// in future GPUs so that user interfaces don't change ; hence users are encouraged
		/// to declare the struct zero in order to handle the assignments of any field
		/// that might be added for later GPUs.
		/// </summary>
		public struct cudaOccDeviceState
		{
			/// <summary/>
			public cudaOccCacheConfig cacheConfig;
		}

		// get the minimum out of two parameters
		private static int __occMin(int lhs, int rhs)
		{
			return rhs < lhs ? rhs : lhs;
		}
		
		// x/y rounding towards +infinity for integers, used to determine # of blocks/warps etc.
		private static int __occDivideRoundUp(int x, int y)
		{
			return (x + (y - 1)) / y;
		}

		// round x towards infinity to the next multiple of y
		private static int __occRoundUp(int x, int y)
		{
			return y * __occDivideRoundUp(x, y);
		}



		
		//////////////////////////////////////////
		//      Architectural Properties        //
		//////////////////////////////////////////

		/*!
		 * Granularity of shared memory allocation
		 */
		private static int cudaOccSMemAllocationGranularity(cudaOccDeviceProp properties)
		{
			switch(properties.computeMajor)
			{
				//case 1:  return 512;
				case 2:  return 128;
				case 3:
				case 5: 
				case 6: return 256;
				default: throw new CudaOccupancyException(cudaOccError.ErrorUnknownDevice);
			}
		}


		/*!
		 * Granularity of register allocation
		 */
		private static int cudaOccRegAllocationGranularity(cudaOccDeviceProp properties, int regsPerThread)
		{
			switch(properties.computeMajor)
			{
				//case 1:  return (properties.minor <= 1) ? 256 : 512;
				case 2:  switch(regsPerThread)
						 {
							case 21:
							case 22:
							case 29:
							case 30:
							case 37:
							case 38:
							case 45:
							case 46:
								return 128;
							default:
								return 64;
						 }
				case 3:
				case 5:
				case 6:  return 256;
				default: throw new CudaOccupancyException(cudaOccError.ErrorUnknownDevice);
			}
		}


		///*!
		// * Granularity of warp allocation
		// */
		//private static int cudaOccWarpAllocationMultiple(cudaOccDeviceProp properties)
		//{
		//	return (properties.major <= 1) ? 2 : 1;
		//}

		/*!
		 * Number of "sides" into which the multiprocessor is partitioned
		 */
		private static int cudaOccSubPartitionsPerMultiprocessor(cudaOccDeviceProp properties)
		{
			switch(properties.computeMajor)
			{
				//case 1:  return 1;
				case 2: return 2;
				case 3: return 4;
				case 5: return 4;
				case 6: return 4;
				default: throw new CudaOccupancyException(cudaOccError.ErrorUnknownDevice);
			}
		}

		/*!
		 * Maximum blocks that can run simultaneously on a multiprocessor
		 */
		private static int cudaOccMaxBlocksPerMultiprocessor(cudaOccDeviceProp properties)
		{
			switch(properties.computeMajor)
			{
				//case 1:  return 8;
				case 2: return 8;
				case 3: return 16;
				case 5: return 32;
				case 6: return 32;
				default: throw new CudaOccupancyException(cudaOccError.ErrorUnknownDevice);
			}
		}

		///*!
		// * Map int to cudaOccCacheConfig
		// */
		//private static cudaOccCacheConfig cudaOccGetCacheConfig(cudaOccDeviceState state)
		//{
		//    switch(state.cacheConfig)
		//    {
		//        case 0:  return cudaOccCacheConfig.PreferNone;
		//        case 1:  return cudaOccCacheConfig.PreferShared;
		//        case 2:  return cudaOccCacheConfig.PreferL1;
		//        case 3:  return cudaOccCacheConfig.PreferEqual;
		//        default: return cudaOccCacheConfig.PreferNone;
		//    }
		//}

		/*!
		 * Shared memory based on config requested by User
		 */
		private static SizeT cudaOccSMemPerMultiprocessor(cudaOccDeviceProp properties, cudaOccCacheConfig cacheConfig)
		{
			SizeT bytes = 0;
			SizeT sharedMemPerMultiprocessorHigh = (int)properties.sharedMemPerMultiprocessor;
			// Fermi and Kepler has shared L1 cache / shared memory, and support cache
			// configuration to trade one for the other. These values are needed to
			// calculate the correct shared memory size for user requested cache
			// configuration.
			//
			SizeT minCacheSize = 16384;
			SizeT maxCacheSize = 49152;
			SizeT cacheAndSharedTotal = sharedMemPerMultiprocessorHigh + minCacheSize;
			SizeT sharedMemPerMultiprocessorLow = cacheAndSharedTotal - maxCacheSize;


			switch (properties.computeMajor)
			{
				case 2:
					// Fermi supports 48KB / 16KB or 16KB / 48KB partitions for shared /
					// L1.
					//
					switch (cacheConfig)
					{
						default:
						case cudaOccCacheConfig.PreferNone:
						case cudaOccCacheConfig.PreferShared:
						case cudaOccCacheConfig.PreferEqual:
							bytes = sharedMemPerMultiprocessorHigh;
							break;
						case cudaOccCacheConfig.PreferL1:
							bytes = sharedMemPerMultiprocessorLow;
							break;
					}
					break;
				case 3:
					// Kepler supports 16KB, 32KB, or 48KB partitions for L1. The rest
					// is shared memory.
					//
					switch (cacheConfig)
					{
						default:
						case cudaOccCacheConfig.PreferNone:
						case cudaOccCacheConfig.PreferShared:
							bytes = sharedMemPerMultiprocessorHigh;
							break;
						case cudaOccCacheConfig.PreferL1:
							bytes = sharedMemPerMultiprocessorLow;
							break;
						case cudaOccCacheConfig.PreferEqual:
							// Equal is the mid-point between high and low. It should be
							// equivalent to low + 16KB.
							//
							bytes = (sharedMemPerMultiprocessorHigh + sharedMemPerMultiprocessorLow) / 2;
							break;
					}
					break;
				case 5:
				case 6:
					// Maxwell and Pascal have dedicated shared memory.
					//
					bytes = sharedMemPerMultiprocessorHigh;
					break;
				default: throw new CudaOccupancyException(cudaOccError.ErrorUnknownDevice);
			}

			return bytes;
		}

		
		/**
		 * Partitioned global caching mode support
		 */
		private static cudaOccPartitionedGCSupport cudaOccPartitionedGlobalCachingModeSupport(cudaOccDeviceProp properties)
		{
			cudaOccPartitionedGCSupport limit = cudaOccPartitionedGCSupport.NotSupported;

			if ((properties.computeMajor == 5 && (properties.computeMinor == 2 || properties.computeMinor == 3)) ||
				properties.computeMajor == 6) 
			{
				limit = cudaOccPartitionedGCSupport.Supported;
			}

			if (properties.computeMajor == 6 && properties.computeMinor == 0) {
				limit = cudaOccPartitionedGCSupport.NotSupported;
			}

			return limit;
		}


		
		///////////////////////////////////////////////
		//            User Input Sanity              //
		///////////////////////////////////////////////

		

		private static cudaOccError cudaOccDevicePropCheck(cudaOccDeviceProp properties)
		{
			// Verify device properties
			//
			// Each of these limits must be a positive number.
			//
			// Compute capacity is checked during the occupancy calculation
			//
			if (properties.maxThreadsPerBlock          <= 0 ||
				properties.maxThreadsPerMultiProcessor <= 0 ||
				properties.regsPerBlock                <= 0 ||
				properties.regsPerMultiprocessor       <= 0 ||
				properties.warpSize                    <= 0 ||
				properties.sharedMemPerBlock           <= 0 ||
				properties.sharedMemPerMultiprocessor  <= 0 ||
				properties.numSms                      <= 0) 
			{
				return cudaOccError.ErrorInvalidInput;
			}

			return cudaOccError.None;
		}

		private static cudaOccError cudaOccFuncAttributesCheck(cudaOccFuncAttributes attributes)
		{
			// Verify function attributes
			//
			if (attributes.maxThreadsPerBlock <= 0 ||
				attributes.numRegs < 0) {            // Compiler may choose not to use
													  // any register (empty kernels,
													  // etc.)
				return cudaOccError.ErrorInvalidInput;
			}

			return cudaOccError.None;
		}

		private static cudaOccError cudaOccDeviceStateCheck(cudaOccDeviceState state)
		{
			// Placeholder
			//

			return cudaOccError.None;
		}

		private static void cudaOccInputCheck(
			cudaOccDeviceProp     properties,
			cudaOccFuncAttributes attributes,
			cudaOccDeviceState    state)
		{
			cudaOccError status = cudaOccError.None;

			status = cudaOccDevicePropCheck(properties);
			if (status != cudaOccError.None)
			{
				throw new CudaOccupancyException(status);
			}

			status = cudaOccFuncAttributesCheck(attributes);
			if (status != cudaOccError.None)
			{
				throw new CudaOccupancyException(status);
			}

			status = cudaOccDeviceStateCheck(state);
			if (status != cudaOccError.None)
			{
				throw new CudaOccupancyException(status);
			}
		}

		///////////////////////////////////////////////
		//    Occupancy calculation Functions        //
		///////////////////////////////////////////////

	
		private static bool cudaOccPartitionedGCForced(cudaOccDeviceProp properties)
		{
			cudaOccPartitionedGCSupport gcSupport = cudaOccPartitionedGlobalCachingModeSupport(properties);

			return gcSupport == cudaOccPartitionedGCSupport.AlwaysOn;
		}
		

		private static cudaOccPartitionedGCConfig cudaOccPartitionedGCExpected(
			cudaOccDeviceProp     properties,
			cudaOccFuncAttributes attributes)
		{
			cudaOccPartitionedGCSupport gcSupport;
			cudaOccPartitionedGCConfig gcConfig;

			gcSupport = cudaOccPartitionedGlobalCachingModeSupport(properties);

			gcConfig = attributes.partitionedGCConfig;

			if (gcSupport == cudaOccPartitionedGCSupport.NotSupported) {
				gcConfig = cudaOccPartitionedGCConfig.Off;
			}

			if (cudaOccPartitionedGCForced(properties)) {
				gcConfig = cudaOccPartitionedGCConfig.On;
			}

			return gcConfig;
		}

		

		// Warp limit
		//
		private static int cudaOccMaxBlocksPerSMWarpsLimit(
			cudaOccPartitionedGCConfig   gcConfig,
			cudaOccDeviceProp     properties,
			cudaOccFuncAttributes attributes,
			int                   blockSize)
		{
			int limit;
			int maxWarpsPerSm;
			int warpsAllocatedPerCTA;
			int maxBlocks;

			if (blockSize > properties.maxThreadsPerBlock) {
				maxBlocks = 0;
			}
			else {
				maxWarpsPerSm = properties.maxThreadsPerMultiProcessor / properties.warpSize;
				warpsAllocatedPerCTA = __occDivideRoundUp(blockSize, properties.warpSize);
				maxBlocks = 0;

				if (gcConfig != cudaOccPartitionedGCConfig.Off) {
					int maxBlocksPerSmPartition;
					int maxWarpsPerSmPartition;

					// If partitioned global caching is on, then a CTA can only use a SM
					// partition (a half SM), and thus a half of the warp slots
					// available per SM
					//
					maxWarpsPerSmPartition  = maxWarpsPerSm / 2;
					maxBlocksPerSmPartition = maxWarpsPerSmPartition / warpsAllocatedPerCTA;
					maxBlocks               = maxBlocksPerSmPartition * 2;
				}
				// On hardware that supports partitioned global caching, each half SM is
				// guaranteed to support at least 32 warps (maximum number of warps of a
				// CTA), so caching will not cause 0 occupancy due to insufficient warp
				// allocation slots.
				//
				else {
					maxBlocks = maxWarpsPerSm / warpsAllocatedPerCTA;
				}
			}

			limit = maxBlocks;

			return limit;
		}
		
		// Shared memory limit
		//
		private static int cudaOccMaxBlocksPerSMSmemLimit(
			cudaOccResult result,
			cudaOccDeviceProp     properties,
			cudaOccFuncAttributes attributes,
			cudaOccDeviceState    state,
			int                   blockSize,
			SizeT                 dynamicSmemSize)
		{
			int allocationGranularity;
			SizeT userSmemPreference;
			SizeT totalSmemUsagePerCTA;
			SizeT smemAllocatedPerCTA;
			SizeT sharedMemPerMultiprocessor;
			int maxBlocks;

			allocationGranularity = cudaOccSMemAllocationGranularity(properties);
			

			// Obtain the user preferred shared memory size. This setting is ignored if
			// user requests more shared memory than preferred.
			//
			userSmemPreference = cudaOccSMemPerMultiprocessor(properties, state.cacheConfig);

			totalSmemUsagePerCTA = attributes.sharedSizeBytes + dynamicSmemSize;
			smemAllocatedPerCTA = __occRoundUp((int)totalSmemUsagePerCTA, (int)allocationGranularity);

			if (smemAllocatedPerCTA > properties.sharedMemPerBlock) {
				maxBlocks = 0;
			}
			else {
				// User requested shared memory limit is used as long as it is greater
				// than the total shared memory used per CTA, i.e. as long as at least
				// one CTA can be launched. Otherwise, the maximum shared memory limit
				// is used instead.
				//
				if (userSmemPreference >= smemAllocatedPerCTA) {
					sharedMemPerMultiprocessor = userSmemPreference;
				}
				else{
					sharedMemPerMultiprocessor = properties.sharedMemPerMultiprocessor;
				}

				if (smemAllocatedPerCTA > 0) {
					maxBlocks = (int)(sharedMemPerMultiprocessor / smemAllocatedPerCTA);
				}
				else {
					maxBlocks = int.MaxValue;
				}
			}

			result.AllocatedSharedMemPerBlock = smemAllocatedPerCTA;

			return maxBlocks;
		}

		

		private static int cudaOccMaxBlocksPerSMRegsLimit(
			ref cudaOccPartitionedGCConfig  gcConfig,
			cudaOccResult         result,
			cudaOccDeviceProp     properties,
			cudaOccFuncAttributes attributes,
			int                   blockSize)
		{
			int allocationGranularity;
			int warpsAllocatedPerCTA;
			int regsAllocatedPerCTA;
			int regsAssumedPerCTA;
			int regsPerWarp;
			int regsAllocatedPerWarp;
			int numSubPartitions;
			int numRegsPerSubPartition;
			int numWarpsPerSubPartition;
			int numWarpsPerSM;
			int maxBlocks;

			allocationGranularity = cudaOccRegAllocationGranularity(
				properties,
				attributes.numRegs);   // Fermi requires special handling of certain register usage

			numSubPartitions = cudaOccSubPartitionsPerMultiprocessor(properties);

			warpsAllocatedPerCTA = __occDivideRoundUp(blockSize, properties.warpSize);

			// GPUs of compute capability 2.x and higher allocate registers to warps
			//
			// Number of regs per warp is regs per thread x warp size, rounded up to
			// register allocation granularity
			//
			regsPerWarp          = attributes.numRegs * properties.warpSize;
			regsAllocatedPerWarp = __occRoundUp(regsPerWarp, allocationGranularity);
			regsAllocatedPerCTA  = regsAllocatedPerWarp * warpsAllocatedPerCTA;

			// Hardware verifies if a launch fits the per-CTA register limit. For
			// historical reasons, the verification logic assumes register
			// allocations are made to all partitions simultaneously. Therefore, to
			// simulate the hardware check, the warp allocation needs to be rounded
			// up to the number of partitions.
			//
			regsAssumedPerCTA = regsAllocatedPerWarp * __occRoundUp(warpsAllocatedPerCTA, numSubPartitions);

			if (properties.regsPerBlock < regsAssumedPerCTA ||   // Hardware check
				properties.regsPerBlock < regsAllocatedPerCTA) { // Software check
				maxBlocks = 0;
			}
			else {
				if (regsAllocatedPerWarp > 0) {
					// Registers are allocated in each sub-partition. The max number
					// of warps that can fit on an SM is equal to the max number of
					// warps per sub-partition x number of sub-partitions.
					//
					numRegsPerSubPartition  = properties.regsPerMultiprocessor / numSubPartitions;
					numWarpsPerSubPartition = numRegsPerSubPartition / regsAllocatedPerWarp;

					maxBlocks = 0;

					if (gcConfig != cudaOccPartitionedGCConfig.Off) {
						int numSubPartitionsPerSmPartition;
						int numWarpsPerSmPartition;
						int maxBlocksPerSmPartition;

						// If partitioned global caching is on, then a CTA can only
						// use a half SM, and thus a half of the registers available
						// per SM
						//
						numSubPartitionsPerSmPartition = numSubPartitions / 2;
						numWarpsPerSmPartition         = numWarpsPerSubPartition * numSubPartitionsPerSmPartition;
						maxBlocksPerSmPartition        = numWarpsPerSmPartition / warpsAllocatedPerCTA;
						maxBlocks                      = maxBlocksPerSmPartition * 2;
					}

					// Try again if partitioned global caching is not enabled, or if
					// the CTA cannot fit on the SM with caching on. In the latter
					// case, the device will automatically turn off caching, except
					// if the device forces it. The user can also override this
					// assumption with PARTITIONED_GC_ON_STRICT to calculate
					// occupancy and launch configuration.
					//
					{
						bool gcOff = (gcConfig == cudaOccPartitionedGCConfig.Off);
						bool zeroOccupancy = (maxBlocks == 0);
						bool cachingForced = (gcConfig == cudaOccPartitionedGCConfig.OnStrict ||
											 cudaOccPartitionedGCForced(properties));

						if (gcOff || (zeroOccupancy && (!cachingForced))) {
							gcConfig = cudaOccPartitionedGCConfig.Off;
							numWarpsPerSM = numWarpsPerSubPartition * numSubPartitions;
							maxBlocks     = numWarpsPerSM / warpsAllocatedPerCTA;
						}
					}
				}
				else {
					maxBlocks = int.MaxValue;
				}
			}


			result.AllocatedRegistersPerBlock = regsAllocatedPerCTA;

			return maxBlocks;
		}

		///////////////////////////////////
		//      API Implementations      //
		///////////////////////////////////


		/// <summary>
		/// Determine the maximum number of CTAs that can be run simultaneously per SM.<para/>
		/// This is equivalent to the calculation done in the CUDA Occupancy Calculator
		/// spreadsheet
		/// </summary>
		/// <param name="result"></param>
		/// <param name="properties"></param>
		/// <param name="attributes"></param>
		/// <param name="state"></param>
		/// <param name="blockSize"></param>
		/// <param name="dynamicSmemSize"></param>
		/// <returns></returns>
		public static void cudaOccMaxActiveBlocksPerMultiprocessor(
			cudaOccResult               result,
			cudaOccDeviceProp     properties,
			cudaOccFuncAttributes attributes,
			cudaOccDeviceState    state,
			int                   blockSize,
			SizeT                 dynamicSmemSize)
		{
			int          ctaLimitWarps   = 0;
			int          ctaLimitBlocks  = 0;
			int          ctaLimitSMem    = 0;
			int          ctaLimitRegs    = 0;
			int          ctaLimit        = 0;
			cudaOccLimitingFactors limitingFactors = 0;
    
			cudaOccPartitionedGCConfig gcConfig = cudaOccPartitionedGCConfig.Off;

			//if (!result || !properties || !attributes || !state || blockSize <= 0) {
			//	return CUDA_OCC_ERROR_INVALID_INPUT;
			//}

			///////////////////////////
			// Check user input
			///////////////////////////

			cudaOccInputCheck(properties, attributes, state);

			///////////////////////////
			// Initialization
			///////////////////////////

			gcConfig = cudaOccPartitionedGCExpected(properties, attributes);

			///////////////////////////
			// Compute occupancy
			///////////////////////////

			// Limits due to registers/SM
			// Also compute if partitioned global caching has to be turned off
			//
			ctaLimitRegs = cudaOccMaxBlocksPerSMRegsLimit(ref gcConfig, result, properties, attributes, blockSize);
			

			// Limits due to warps/SM
			//
			ctaLimitWarps = cudaOccMaxBlocksPerSMWarpsLimit(gcConfig, properties, attributes, blockSize);
			

			// Limits due to blocks/SM
			//
			ctaLimitBlocks = cudaOccMaxBlocksPerMultiprocessor(properties);

			// Limits due to shared memory/SM
			//
			ctaLimitSMem = cudaOccMaxBlocksPerSMSmemLimit(result, properties, attributes, state, blockSize, dynamicSmemSize);
			

			///////////////////////////
			// Overall occupancy
			///////////////////////////

			// Overall limit is min() of limits due to above reasons
			//
			ctaLimit = __occMin(ctaLimitRegs, __occMin(ctaLimitSMem, __occMin(ctaLimitWarps, ctaLimitBlocks)));

			// Fill in the return values
			//
			// Determine occupancy limiting factors
			//
			if (ctaLimit == ctaLimitWarps) {
				limitingFactors |= cudaOccLimitingFactors.Warps;
			}
			if (ctaLimit == ctaLimitRegs) {
				limitingFactors |= cudaOccLimitingFactors.Registers;
			}
			if (ctaLimit == ctaLimitSMem) {
				limitingFactors |= cudaOccLimitingFactors.SharedMemory;
			}
			if (ctaLimit == ctaLimitBlocks) {
				limitingFactors |= cudaOccLimitingFactors.Blocks;
			}
			result.LimitingFactors = limitingFactors;

			result.BlockLimitRegs      = ctaLimitRegs;
			result.BlockLimitSharedMem = ctaLimitSMem;
			result.BlockLimitWarps     = ctaLimitWarps;
			result.BlockLimitBlocks    = ctaLimitBlocks;
			result.partitionedGCConfig = gcConfig;

			// Final occupancy
			result.ActiveBlocksPerMultiProcessor = ctaLimit;

		}

		
		/// <summary>
		/// 
		/// </summary>
		/// <param name="minGridSize"></param>
		/// <param name="blockSize"></param>
		/// <param name="properties"></param>
		/// <param name="attributes"></param>
		/// <param name="state"></param>
		/// <param name="blockSizeToDynamicSMemSize"></param>
		/// <param name="dynamicSMemSize"></param>
		public static void cudaOccMaxPotentialOccupancyBlockSize(
			ref int                         minGridSize,
			ref int                         blockSize,
			cudaOccDeviceProp     properties,
			cudaOccFuncAttributes attributes,
			cudaOccDeviceState    state,
			del_blockSizeToDynamicSMemSize blockSizeToDynamicSMemSize,
			SizeT                       dynamicSMemSize)
		{
			cudaOccResult result = new cudaOccResult();

			// Limits
			int occupancyLimit;
			int granularity;
			int blockSizeLimit;

			// Recorded maximum
			int maxBlockSize = 0;
			int numBlocks    = 0;
			int maxOccupancy = 0;

			// Temporary
			int blockSizeToTryAligned;
			int blockSizeToTry;
			int blockSizeLimitAligned;
			int occupancyInBlocks;
			int occupancyInThreads;

			///////////////////////////
			// Check user input
			///////////////////////////

			//if (!minGridSize || !blockSize || !properties || !attributes || !state) {
			//	return CUDA_OCC_ERROR_INVALID_INPUT;
			//}

			cudaOccInputCheck(properties, attributes, state);

			/////////////////////////////////////////////////////////////////////////////////
			// Try each block size, and pick the block size with maximum occupancy
			/////////////////////////////////////////////////////////////////////////////////

			occupancyLimit = properties.maxThreadsPerMultiProcessor;
			granularity    = properties.warpSize;

			blockSizeLimit        = __occMin(properties.maxThreadsPerBlock, attributes.maxThreadsPerBlock);
			blockSizeLimitAligned = __occRoundUp(blockSizeLimit, granularity);

			for (blockSizeToTryAligned = blockSizeLimitAligned; blockSizeToTryAligned > 0; blockSizeToTryAligned -= granularity) {
				blockSizeToTry = __occMin(blockSizeLimit, blockSizeToTryAligned);

				// Ignore dynamicSMemSize if the user provides a mapping
				//
				if (blockSizeToDynamicSMemSize != null) {
					dynamicSMemSize = blockSizeToDynamicSMemSize(blockSizeToTry);
				}

				cudaOccMaxActiveBlocksPerMultiprocessor(
					result,
					properties,
					attributes,
					state,
					blockSizeToTry,
					dynamicSMemSize);

				//if (status != CUDA_OCC_SUCCESS) {
				//	return status;
				//}

				occupancyInBlocks = result.ActiveBlocksPerMultiProcessor;
				occupancyInThreads = blockSizeToTry * occupancyInBlocks;

				if (occupancyInThreads > maxOccupancy) {
					maxBlockSize = blockSizeToTry;
					numBlocks    = occupancyInBlocks;
					maxOccupancy = occupancyInThreads;
				}

				// Early out if we have reached the maximum
				//
				if (occupancyLimit == maxOccupancy) {
					break;
				}
			}

			///////////////////////////
			// Return best available
			///////////////////////////

			// Suggested min grid size to achieve a full machine launch
			//
			minGridSize = numBlocks * properties.numSms;
			blockSize = maxBlockSize;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="minGridSize"></param>
		/// <param name="blockSize"></param>
		/// <param name="properties"></param>
		/// <param name="attributes"></param>
		/// <param name="state"></param>
		/// <param name="dynamicSMemSize"></param>
		public static void cudaOccMaxPotentialOccupancyBlockSize(
			ref int minGridSize,
			ref int blockSize,
			cudaOccDeviceProp properties,
			cudaOccFuncAttributes attributes,
			cudaOccDeviceState state,
			SizeT dynamicSMemSize)
		{
			cudaOccMaxPotentialOccupancyBlockSize(ref minGridSize, ref blockSize, properties, attributes, state, null, dynamicSMemSize);
		}


		///// <summary>
		///// Determine the maximum number of CTAs that can be run simultaneously per SM.<para/>
		///// This is equivalent to the calculation done in the CUDA Occupancy Calculator
		///// spreadsheet
		///// </summary>
		///// <param name="properties"></param>
		///// <param name="attributes"></param>
		///// <param name="blockSize"></param>
		///// <param name="dynamic_smem_bytes"></param>
		///// <param name="state"></param>
		///// <returns></returns>
		//public static cudaOccResult cudaOccMaxActiveBlocksPerMultiprocessor(
		//	cudaOccDeviceProp properties,
		//	cudaOccFuncAttributes attributes,
		//	int blockSize,
		//	SizeT dynamic_smem_bytes,
		//	cudaOccDeviceState state)
		//{
		//	int regAllocationUnit = 0, warpAllocationMultiple = 0, maxBlocksPerSM=0;
		//	int ctaLimitWarps = 0, ctaLimitBlocks = 0, smemPerCTA = 0, smemBytes = 0, smemAllocationUnit = 0;
		//	int cacheConfigSMem = 0, sharedMemPerMultiprocessor = 0, ctaLimitRegs = 0, regsPerCTA=0;
		//	int regsPerWarp = 0, numSides = 0, numRegsPerSide = 0, ctaLimit=0;
		//	int maxWarpsPerSm = 0, warpsPerCTA = 0, ctaLimitSMem=0;
		//	cudaOccLimitingFactors limitingFactors = 0;
		//	cudaOccResult result = new cudaOccResult();

		//	if(properties == null || attributes == null || blockSize <= 0)
		//	{
		//		throw new CudaOccupancyException(cudaOccError.ErrorInvalidInput);
		//	}

		//	//////////////////////////////////////////
		//	// Limits due to warps/SM or blocks/SM
		//	//////////////////////////////////////////
		//	CudaOccupancyException.CheckZero(properties.warpSize);
		//	maxWarpsPerSm   = properties.maxThreadsPerMultiProcessor / properties.warpSize;
		//	warpAllocationMultiple = cudaOccWarpAllocationMultiple(properties);

		//	CudaOccupancyException.CheckZero(warpAllocationMultiple);
		//	warpsPerCTA = __occRoundUp(__occDivideRoundUp(blockSize, properties.warpSize), warpAllocationMultiple);

		//	maxBlocksPerSM  = cudaOccMaxBlocksPerMultiprocessor(properties);

		//	// Calc limits
		//	CudaOccupancyException.CheckZero(warpsPerCTA);
		//	ctaLimitWarps  = (blockSize <= properties.maxThreadsPerBlock) ? maxWarpsPerSm / warpsPerCTA : 0;
		//	ctaLimitBlocks = maxBlocksPerSM;

		//	//////////////////////////////////////////
		//	// Limits due to shared memory/SM
		//	//////////////////////////////////////////
		//	smemAllocationUnit     = cudaOccSMemAllocationUnit(properties);
		//	smemBytes  = (int)(attributes.sharedSizeBytes + dynamic_smem_bytes);
		//	CudaOccupancyException.CheckZero(smemAllocationUnit);
		//	smemPerCTA = __occRoundUp(smemBytes, smemAllocationUnit);

		//	// Calc limit
		//	cacheConfigSMem = cudaOccSMemPerMultiprocessor(properties,state.cacheConfig);

		//	// sharedMemoryPerMultiprocessor is by default limit set in hardware but user requested shared memory
		//	// limit is used instead if it is greater than total shared memory used by function .
		//	sharedMemPerMultiprocessor = (cacheConfigSMem >= smemPerCTA)
		//		? cacheConfigSMem
		//		: (int)properties.sharedMemPerMultiprocessor;
		//	// Limit on blocks launched should be calculated with shared memory per SM but total shared memory
		//	// used by function should be limited by shared memory per block
		//	ctaLimitSMem = 0;
		//	if(properties.sharedMemPerBlock >= (SizeT)smemPerCTA)
		//	{
		//		ctaLimitSMem = smemPerCTA > 0 ? sharedMemPerMultiprocessor / smemPerCTA : maxBlocksPerSM;
		//	}

		//	//////////////////////////////////////////
		//	// Limits due to registers/SM
		//	//////////////////////////////////////////
		//	regAllocationUnit      = cudaOccRegAllocationUnit(properties, attributes.numRegs);
		//	CudaOccupancyException.CheckZero(regAllocationUnit);

		//	// Calc limit
		//	ctaLimitRegs = 0;
		//	if(properties.computeMajor <= 1)
		//	{
		//		// GPUs of compute capability 1.x allocate registers to CTAs
		//		// Number of regs per block is regs per thread times number of warps times warp size, rounded up to allocation unit
		//		regsPerCTA = __occRoundUp(attributes.numRegs * properties.warpSize * warpsPerCTA, regAllocationUnit);
		//		ctaLimitRegs = regsPerCTA > 0 ? properties.regsPerMultiprocessor / regsPerCTA : maxBlocksPerSM;
		//	}
		//	else
		//	{
		//		// GPUs of compute capability 2.x and higher allocate registers to warps
		//		// Number of regs per warp is regs per thread times number of warps times warp size, rounded up to allocation unit
		//		regsPerWarp = __occRoundUp(attributes.numRegs * properties.warpSize, regAllocationUnit);
		//		regsPerCTA = regsPerWarp * warpsPerCTA;
		//		if(properties.regsPerBlock >= regsPerCTA)
		//		{
		//			numSides = cudaOccSidesPerMultiprocessor(properties);
		//			CudaOccupancyException.CheckZero(numSides);
		//			numRegsPerSide = properties.regsPerMultiprocessor / numSides;
		//			ctaLimitRegs = regsPerWarp > 0 ? ((numRegsPerSide / regsPerWarp) * numSides) / warpsPerCTA : maxBlocksPerSM;
		//		}
		//	}

		//	//////////////////////////////////////////
		//	// Overall limit is min() of limits due to above reasons
		//	//////////////////////////////////////////
		//	ctaLimit = __occMin(ctaLimitRegs, __occMin(ctaLimitSMem, __occMin(ctaLimitWarps, ctaLimitBlocks)));
		//	// Determine occupancy limiting factors
			
			
		//	result.ActiveBlocksPerMultiProcessor = ctaLimit;

		//	if(ctaLimit==ctaLimitWarps)
		//	{
		//		limitingFactors |= cudaOccLimitingFactors.Warps;
		//	}
		//	if(ctaLimit==ctaLimitRegs && regsPerCTA > 0)
		//	{
		//		limitingFactors |= cudaOccLimitingFactors.Registers;
		//	}
		//	if(ctaLimit==ctaLimitSMem && smemPerCTA > 0)
		//	{
		//		limitingFactors |= cudaOccLimitingFactors.SharedMemory;
		//	}
		//	if(ctaLimit==ctaLimitBlocks)
		//	{
		//		limitingFactors |= cudaOccLimitingFactors.Blocks;
		//	}
		//	result.LimitingFactors = limitingFactors;

		//	result.BlockLimitRegs = ctaLimitRegs;
		//	result.BlockLimitSharedMem = ctaLimitSMem;
		//	result.BlockLimitWarps = ctaLimitWarps;
		//	result.BlockLimitBlocks = ctaLimitBlocks;

		//	result.AllocatedRegistersPerBlock = regsPerCTA;
		//	result.AllocatedSharedMemPerBlock = smemPerCTA;

		//	result.ActiveWarpsPerMultiProcessor = ctaLimit * ((int)Math.Ceiling(blockSize / (double)properties.warpSize));
		//	result.ActiceThreadsPerMultiProcessor = result.ActiveWarpsPerMultiProcessor * properties.warpSize;
		//	result.OccupancyOfEachMultiProcessor = (int)Math.Round(result.ActiveWarpsPerMultiProcessor / (double)maxWarpsPerSm * 100);
		//	return result;
		//}

		/// <summary>
		/// A function to convert from block size to dynamic shared memory size.<para/>
		/// e.g.:
		/// If no dynamic shared memory is used: x => 0<para/>
		/// If 4 bytes shared memory per thread is used: x = 4 * x
		/// </summary>
		/// <param name="aBlockSize">block size</param>
		/// <returns>size of dynamic shared memory</returns>
		public delegate SizeT del_blockSizeToDynamicSMemSize(int aBlockSize);
		

		///// <summary>
		///// Determine the potential block size that allows maximum number of CTAs that can run on multiprocessor simultaneously 
		///// </summary>
		///// <param name="properties"></param>
		///// <param name="kernel"></param>
		///// <param name="state"></param>
		///// <param name="blockSizeToSMem">
		///// A function to convert from block size to dynamic shared memory size.<para/>
		///// e.g.:
		///// If no dynamic shared memory is used: x => 0<para/>
		///// If 4 bytes shared memory per thread is used: x = 4 * x</param>
		///// <returns>maxBlockSize</returns>
		//public static int cudaOccMaxPotentialOccupancyBlockSize(
		//	CudaDeviceProperties properties,
		//	CudaKernel kernel,
		//	cudaOccDeviceState state,
		//	del_blockSizeToDynamicSMemSize blockSizeToSMem)
		//{
		//	cudaOccDeviceProp props = new cudaOccDeviceProp(properties);
		//	cudaOccFuncAttributes attributes = new cudaOccFuncAttributes(kernel);
		//	return cudaOccMaxPotentialOccupancyBlockSize(props, attributes, state, blockSizeToSMem);
		//}

		///// <summary>
		///// Determine the potential block size that allows maximum number of CTAs that can run on multiprocessor simultaneously 
		///// </summary>
		///// <param name="properties"></param>
		///// <param name="attributes"></param>
		///// <param name="state"></param>
		///// <param name="blockSizeToSMem">
		///// A function to convert from block size to dynamic shared memory size.<para/>
		///// e.g.:
		///// If no dynamic shared memory is used: x => 0<para/>
		///// If 4 bytes shared memory per thread is used: x = 4 * x</param>
		///// <returns>maxBlockSize</returns>
		//public static int cudaOccMaxPotentialOccupancyBlockSize(
		//	cudaOccDeviceProp properties,
		//	cudaOccFuncAttributes attributes,
		//	cudaOccDeviceState state,
		//	del_blockSizeToDynamicSMemSize blockSizeToSMem)
		//{
		//	int maxOccupancy       = properties.maxThreadsPerMultiProcessor;
		//	int largestBlockSize   = __occMin(properties.maxThreadsPerBlock, attributes.maxThreadsPerBlock);
		//	int granularity        = properties.warpSize;
		//	int maxBlockSize  = 0;
		//	int blockSize     = 0;
		//	int highestOccupancy   = 0;

		//	for(blockSize = largestBlockSize; blockSize > 0; blockSize -= granularity)
		//	{
		//		cudaOccResult res = cudaOccMaxActiveBlocksPerMultiprocessor(properties, attributes, blockSize, blockSizeToSMem(blockSize), state);
		//		int occupancy = res.ActiveBlocksPerMultiProcessor;
		//		occupancy = blockSize*occupancy;

		//		if(occupancy > highestOccupancy)
		//		{
		//			maxBlockSize = blockSize;
		//			highestOccupancy = occupancy;
		//		}

		//		// can not get higher occupancy
		//		if(highestOccupancy == maxOccupancy)
		//			break;
		//	}

		//	return maxBlockSize;
		//}




	}
}
