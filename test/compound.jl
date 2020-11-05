using Random, Test, HDF5

import HDF5.datatype
import Base.convert
import Base.unsafe_convert

struct foo
    a::Float64
    b::String
    c::String
    d::Array{ComplexF64,2}
    e::Array{Int64,1}
end

struct foo_hdf5
    a::Float64
    b::Cstring
    c::NTuple{20, UInt8}
    d::NTuple{9, ComplexF64}
    e::HDF5.hvl_t
end

function HDF5.pack(x::foo)
    foo_hdf5(x.a,
            Base.unsafe_convert(Cstring, x.b),
            ntuple(i -> i <= ncodeunits(x.c) ? codeunit(x.c, i) : '\0', 20),
            ntuple(i -> x.d[i], length(x.d)),
            HDF5.hvl_t(length(x.e), pointer(x.e))
            )
end

function datatype(::Type{foo})
    dtype = HDF5.h5t_create(HDF5.H5T_COMPOUND, sizeof(foo_hdf5))
    HDF5.h5t_insert(dtype, "a", fieldoffset(foo_hdf5, 1), datatype(Float64))

    vlenstr_dtype = HDF5.h5t_copy(HDF5.H5T_C_S1)
    HDF5.h5t_set_size(vlenstr_dtype, HDF5.H5T_VARIABLE)
    HDF5.h5t_set_cset(vlenstr_dtype, HDF5.H5T_CSET_UTF8)
    HDF5.h5t_insert(dtype, "b", fieldoffset(foo_hdf5, 2), vlenstr_dtype)

    fixedstr_dtype = HDF5.h5t_copy(HDF5.H5T_C_S1)
    HDF5.h5t_set_size(fixedstr_dtype, 20 * sizeof(UInt8))
    HDF5.h5t_set_cset(fixedstr_dtype, HDF5.H5T_CSET_UTF8)
    HDF5.h5t_set_strpad(fixedstr_dtype, HDF5.H5T_STR_NULLPAD)
    HDF5.h5t_insert(dtype, "c", fieldoffset(foo_hdf5, 3), fixedstr_dtype)

    hsz = HDF5.hsize_t[3,3]
    array_dtype = HDF5.h5t_array_create(datatype(ComplexF64).id, 2, hsz)
    HDF5.h5t_insert(dtype, "d", fieldoffset(foo_hdf5, 4), array_dtype)

    vlen_dtype = HDF5.h5t_vlen_create(datatype(Int64))
    HDF5.h5t_insert(dtype, "e", fieldoffset(foo_hdf5, 5), vlen_dtype)

    HDF5.Datatype(dtype)
end

struct bar
    a::Vector{String}
end

struct bar_hdf5
    a::NTuple{2, NTuple{20, UInt8}}
end

function datatype(::Type{bar})
    dtype = HDF5.h5t_create(HDF5.H5T_COMPOUND, sizeof(bar_hdf5))

    fixedstr_dtype = HDF5.h5t_copy(HDF5.H5T_C_S1)
    HDF5.h5t_set_size(fixedstr_dtype, 20 * sizeof(UInt8))
    HDF5.h5t_set_cset(fixedstr_dtype, HDF5.H5T_CSET_UTF8)

    hsz = HDF5.hsize_t[2]
    array_dtype = HDF5.h5t_array_create(fixedstr_dtype, 1, hsz)

    HDF5.h5t_insert(dtype, "a", fieldoffset(bar_hdf5, 1), array_dtype)

    HDF5.Datatype(dtype)
end

function HDF5.pack(x::bar)
    bar_hdf5(ntuple(i -> ntuple(j -> j <= ncodeunits(x.a[i]) ? codeunit(x.a[i], j) : '\0', 20), 2))
end

@testset "compound" begin
    N = 10
    v = [foo(rand(),
            randstring(rand(10:100)),
            randstring(10),
            rand(ComplexF64, 3,3),
            rand(1:10, rand(10:100))
            )
        for _ in 1:N]

    v[1] = foo(1.0,
              "uniçº∂e",
              "uniçº∂e",
              rand(ComplexF64, 3,3),
              rand(1:10, rand(10:100)))

    w = [bar(["uniçº∂e", "hello"])]

    fn = tempname()
    h5open(fn, "w") do h5f
        h5f["foo"] = v
        h5f["bar"] = w
    end

    v_read = h5read(fn, "foo")
    for field in (:a, :b, :c, :d, :e)
        f = x -> getfield(x, field)
        @test f.(v) == f.(v_read)
    end

    w_read = h5read(fn, "bar")
    for field in (:a,)
        f = x -> getfield(x, field)
        @test f.(w) == f.(w_read)
    end

    T = NamedTuple{(:a, :b, :c, :d, :e, :f), Tuple{Int, Int, Int, Int, Int, Cstring}}
    TT = NamedTuple{(:a, :b, :c, :d, :e, :f), Tuple{Int, Int, Int, Int, Int, T}}
    TTT = NamedTuple{(:a, :b, :c, :d, :e, :f), Tuple{Int, Int, Int, Int, Int, TT}}
    TTTT = NamedTuple{(:a, :b, :c, :d, :e, :f), Tuple{Int, Int, Int, Int, Int, TTT}}

    @test HDF5.do_reclaim(TTTT) == true
    @test HDF5.do_normalize(TTTT) == true

    T = NamedTuple{(:a, :b, :c, :d, :e, :f), Tuple{Int, Int, Int, Int, Int, HDF5.FixedArray}}
    TT = NamedTuple{(:a, :b, :c, :d, :e, :f), Tuple{Int, Int, Int, Int, Int, T}}
    TTT = NamedTuple{(:a, :b, :c, :d, :e, :f), Tuple{Int, Int, Int, Int, Int, TT}}
    TTTT = NamedTuple{(:a, :b, :c, :d, :e, :f), Tuple{Int, Int, Int, Int, Int, TTT}}

    @test HDF5.do_reclaim(TTTT) == false
    @test HDF5.do_normalize(TTTT) == true
end
