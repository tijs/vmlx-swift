import Foundation
import MLX
import MLXFast
#if canImport(CryptoKit)
import CryptoKit
#else
import Crypto
#endif

/// B1/top-k diagnostic. Only independently mapped expert allocations may be
/// supplied by the selected-bank owner; array views of a whole bank defeat the
/// working-set goal even when their visible shapes match. No factory enables it.
final class JANGHSelectedExpertDecode {
    struct Hidden {
        let values: MLXArray
        fileprivate let expertIDs: [UInt32]
        fileprivate let kernelIdentity: String
        fileprivate let ownerIdentity: UUID
    }
    let identity: String
    private let ownerIdentity: UUID
    private let gateModule: String, upModule: String, downModule: String
    private let gateBits: Int, upBits: Int, downBits: Int
    private let inputRotation: JANGHFormatContract.Rotation
    private let downRotation: JANGHFormatContract.Rotation
    private let gu: MLXFast.MLXFastKernel, down: MLXFast.MLXFastKernel
    private let rotation = JANGHRowRotation()
    /// Include the output: gate/up needs 5 + 2R inputs and one output.
    /// Metal's 31 buffer slots therefore admit at most twelve routes.
    static let maxRoutes = 12
    let routes: Int

    init(owner: JANGHExpertMappedBanks, gateModule: String, upModule: String, downModule: String, routes: Int = 8) throws {
        guard (1 ... Self.maxRoutes).contains(routes) else {
            throw JANGHFormatContract.ValidationError.invalid("selected decode route count out of range")
        }
        self.routes = routes
        let contract = owner.contract
        ownerIdentity = owner.identity
        guard let g = contract.projections[gateModule], let u = contract.projections[upModule],
            let d = contract.projections[downModule], g.rotation == u.rotation,
            let gb = contract.codebooks[g.bits], let ub = contract.codebooks[u.bits],
            let db = contract.codebooks[d.bits]
        else { throw JANGHFormatContract.ValidationError.invalid("incompatible selected decode projections") }
        self.gateModule = gateModule; self.upModule = upModule; self.downModule = downModule
        gateBits = g.bits; upBits = u.bits; downBits = d.bits
        inputRotation = g.rotation; downRotation = d.rotation
        let descriptor = "selected-b1-top\(routes)-v2|\(gateModule)|\(upModule)|\(downModule)|\(g.bits)|\(u.bits)|\(d.bits)|\(g.rotation)|\(d.rotation)|\(gb.alpha.bitPattern)|\(gb.beta.bitPattern)|\(ub.alpha.bitPattern)|\(ub.beta.bitPattern)|\(db.alpha.bitPattern)|\(db.beta.bitPattern)"
        identity = SHA256.hash(data: Data(descriptor.utf8)).map { String(format: "%02x", $0) }.joined()
        let gateDot = JANGHDecodeQDot.source(bits:g.bits,packed:"gateBank",rowBase:"size_t(row0 + r) * WG",columnBase:"block + lane * 16u",values:"values",accumulator:"pg",alpha:Float(gb.alpha),beta:Float(gb.beta))
        let upDot = JANGHDecodeQDot.source(bits:u.bits,packed:"upBank",rowBase:"size_t(row0 + r) * WU",columnBase:"block + lane * 16u",values:"values",accumulator:"pu",alpha:Float(ub.alpha),beta:Float(ub.beta))
        func bankChoice(_ name: String) -> String {
            (0..<(routes - 1)).map { "route == \($0) ? \(name)\($0) : " }.joined() + "\(name)\(routes - 1)"
        }
        let gateNames = (0..<routes).map { "gate\($0)" }, upNames = (0..<routes).map { "up\($0)" }
        // 5 + 2R input buffers plus one output; no full bank or per-route copy.
        gu = MLXFast.metalKernel(name:"jangh_selected_gu_"+identity,
            inputNames:["x"] + gateNames + upNames + ["gs","us","ids","limitValue"],outputNames:["out"],source:"""
            uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
            uint route = threadgroup_position_in_grid.z;
            uint row0 = threadgroup_position_in_grid.y * ROWS + sg * 4u;
            uint expert = ids[route];
            if (expert >= EXPERTS) {
                if (lane == 0) for (uint r=0;r<4;++r)
                    if (row0+r<N) out[size_t(route)*N+row0+r]=as_type<float>(0x7fc00000u);
                return;
            }
            const device uint* gateBank = \(bankChoice("gate"));
            const device uint* upBank = \(bankChoice("up"));
            float ag[4]={0,0,0,0}, au[4]={0,0,0,0};
            for (uint block=0;block<K;block+=512u) {
                float values[16];
                _Pragma("clang loop unroll(full)")
                for (uint i=0;i<16;++i) {
                    uint col=block+lane*16u+i;
                    values[i]=col<K ? float(x[col]) : 0.0f;
                }
                _Pragma("clang loop unroll(full)")
                for (uint r=0;r<4;++r) {
                    if (row0+r>=N) continue;
                    float pg=0.0f, pu=0.0f;
                    if (block+lane*16u<K) {
                        \(gateDot)
                        \(upDot)
                    }
                    ag[r]+=pg; au[r]+=pu;
                }
            }
            threadgroup float hb[32];
            for (uint r=0;r<4;++r) {
                float g=simd_sum(ag[r]), u=simd_sum(au[r]);
                if (lane==0 && row0+r<N) {
                    g*=float(gs[size_t(expert)*N+row0+r]);
                    u*=float(us[size_t(expert)*N+row0+r]);
                    float limit=limitValue[0];
                    if (limit>0.0f) { g=metal::min(g,limit); u=metal::clamp(u,-limit,limit); }
                    float h=(g/(1.0f+metal::fast::exp(-g)))*u;
                    if (ROT_OUT) hb[sg*4u+r]=h;
                    else out[size_t(route)*N+row0+r]=h;
                }
            }
            if (ROT_OUT) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg==0) {
                    float value=hb[lane];
                    for (ushort stage=1;stage<32;stage<<=1) {
                        float other=simd_shuffle_xor(value,stage);
                        value=(lane&stage) ? other-value : value+other;
                    }
                    out[size_t(route)*N+threadgroup_position_in_grid.y*32u+lane]=value*0.17677669529663687f;
                }
            }
            """,ensureRowContiguous:false)
        let downDot = JANGHDecodeQDot.source(bits:d.bits,packed:"downBank",rowBase:"size_t(row0+r)*WORDS",columnBase:"block+lane*16u",values:"values",accumulator:"accum[r]",alpha:Float(db.alpha),beta:Float(db.beta))
        // 4 + R inputs plus output; routes are accumulated sequentially in F32.
        down = MLXFast.metalKernel(name:"jangh_selected_down_"+identity,
            inputNames:["hidden"]+(0..<routes).map { "down\($0)" }+["scales","ids","scores"],outputNames:["out"],source:"""
            uint lane=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
            uint row0=threadgroup_position_in_grid.y*8u+sg*4u;
            float total[4]={0,0,0,0};
            for (uint route=0;route<\(routes)u;++route) {
                uint expert=ids[route];
                if (expert>=EXPERTS) {
                    if (lane==0) for (uint r=0;r<4;++r)
                        if (row0+r<N) out[row0+r]=T(as_type<float>(0x7fc00000u));
                    return;
                }
                const device uint* downBank = \(bankChoice("down"));
                float accum[4]={0,0,0,0};
                for (uint block=0;block<H;block+=512u) {
                    if (block+lane*16u>=H) continue;
                    float values[16];
                    _Pragma("clang loop unroll(full)")
                    for (uint i=0;i<16;++i) values[i]=hidden[size_t(route)*H+block+lane*16u+i];
                    _Pragma("clang loop unroll(full)")
                    for (uint r=0;r<4;++r) {
                        if (row0+r>=N) continue;
                        \(downDot)
                    }
                }
                for (uint r=0;r<4;++r) {
                    float dot=simd_sum(accum[r]);
                    if (row0+r<N) total[r]+=scores[route]*dot*float(scales[size_t(expert)*N+row0+r]);
                }
            }
            if (lane==0) for (uint r=0;r<4;++r) if (row0+r<N) out[row0+r]=T(total[r]);
            """,ensureRowContiguous:false)
    }

    func prepareInput(_ input: MLXArray) throws -> MLXArray {
        try validateInput(input)
        return inputRotation == .hadamard32 ? try rotation(input,outputDType:.float32) : input
    }
    private func validateInput(_ input: MLXArray) throws {
        guard input.ndim==2, input.dim(0)==1, input.dim(1)>0, input.dim(1).isMultiple(of:32),
            [.float16,.bfloat16,.float32].contains(input.dtype)
        else { throw JANGHFormatContract.ValidationError.invalid("selected decode requires B1 input") }
    }
    private func validate(_ selection: JANGHExpertMappedBanks.Selection, module: String, inputWidth: Int, bits: Int) throws -> (Int,Int,Int) {
        let product=inputWidth.multipliedReportingOverflow(by:bits)
        guard selection.ownerIdentity==ownerIdentity, selection.module==module, selection.expertIDs.count==routes, selection.packed.count==routes,
            !product.overflow, product.partialValue<=Int(UInt32.max), inputWidth>0, inputWidth.isMultiple(of:32),
            selection.scales.ndim==2, selection.scales.dtype == .float16,
            selection.scales.dim(0)>0, selection.scales.dim(0)<=Int(UInt32.max),
            selection.scales.dim(1)>0, selection.scales.dim(1)<=Int(UInt32.max)-31
        else { throw JANGHFormatContract.ValidationError.invalid("invalid selected decode metadata") }
        let e=selection.scales.dim(0), n=selection.scales.dim(1), words=product.partialValue/32
        guard selection.expertIDs.allSatisfy({ UInt64($0)<UInt64(e) }) else {
            throw JANGHFormatContract.ValidationError.invalid("invalid selected expert ID")
        }
        try JANGHBankLayout.requireReadyRowContiguous(selection.scales,role:"selected scales")
        for bank in selection.packed {
            guard bank.dtype == .uint32, bank.shape == [1,n,words] else {
                throw JANGHFormatContract.ValidationError.invalid("invalid selected expert shape")
            }
            try JANGHBankLayout.requireReadyRowContiguous(bank,role:"selected expert")
        }
        return (e,n,words)
    }
    func activatePreparedInput(_ input: MLXArray, gate: JANGHExpertMappedBanks.Selection,
        up: JANGHExpertMappedBanks.Selection, limit: Float?) throws -> Hidden {
        try validateInput(input)
        let (e,n,wg)=try validate(gate,module:gateModule,inputWidth:input.dim(1),bits:gateBits)
        let (ue,un,wu)=try validate(up,module:upModule,inputWidth:input.dim(1),bits:upBits)
        guard gate.expertIDs==up.expertIDs,e==ue,n==un,
            downRotation != .hadamard32 || n.isMultiple(of:32),
            limit == nil || (limit!.isFinite && limit!>0)
        else { throw JANGHFormatContract.ValidationError.invalid("incompatible selected gate/up") }
        let rotated=downRotation == .hadamard32, rows=rotated ? 32 : 8, threads=rotated ? 256 : 64
        let values=gu([contiguous(input)]+gate.packed+up.packed+[gate.scales,up.scales,MLXArray(gate.expertIDs),MLXArray([limit ?? 0])],
            template:[("K",input.dim(1)),("N",n),("EXPERTS",e),("ROWS",rows),("WG",wg),("WU",wu),("ROT_OUT",rotated)],
            grid:(threads,(n+rows-1)/rows,routes),threadGroup:(threads,1,1),outputShapes:[[routes,n]],outputDTypes:[.float32])[0]
        return Hidden(values:values,expertIDs:gate.expertIDs,kernelIdentity:identity,ownerIdentity:ownerIdentity)
    }
    func projectPreparedHidden(_ hidden: Hidden, down selection: JANGHExpertMappedBanks.Selection,
        scores: MLXArray, outputDType: DType) throws -> MLXArray {
        guard hidden.ownerIdentity==ownerIdentity, hidden.kernelIdentity==identity, hidden.values.ndim==2, hidden.values.dim(0)==routes,
            hidden.values.dtype == .float32, hidden.expertIDs==selection.expertIDs,
            scores.shape==[1,routes],scores.dtype == .float32,[.float16,.bfloat16,.float32].contains(outputDType)
        else { throw JANGHFormatContract.ValidationError.invalid("invalid selected hidden/route contract") }
        let (e,n,words)=try validate(selection,module:downModule,inputWidth:hidden.values.dim(1),bits:downBits)
        return down([contiguous(hidden.values)]+selection.packed+[selection.scales,MLXArray(selection.expertIDs),contiguous(scores)],
            template:[("H",hidden.values.dim(1)),("N",n),("EXPERTS",e),("WORDS",words),("T",outputDType)],
            grid:(64,(n+7)/8,1),threadGroup:(64,1,1),outputShapes:[[1,n]],outputDTypes:[outputDType])[0]
    }
}
