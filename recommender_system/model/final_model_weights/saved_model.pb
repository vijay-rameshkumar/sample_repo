??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.9.02v2.9.0-rc2-42-g8a20d54a3c18??
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

: *
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:@ *
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*&
shared_nameAdam/dense_1/kernel/v
?
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	?@*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/embedding_1/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:8@*.
shared_nameAdam/embedding_1/embeddings/v
?
1Adam/embedding_1/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_1/embeddings/v*
_output_shapes

:8@*
dtype0
?
Adam/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*,
shared_nameAdam/embedding/embeddings/v
?
/Adam/embedding/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/v*
_output_shapes
:	?@*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

: *
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:@ *
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*&
shared_nameAdam/dense_1/kernel/m
?
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	?@*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/embedding_1/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:8@*.
shared_nameAdam/embedding_1/embeddings/m
?
1Adam/embedding_1/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_1/embeddings/m*
_output_shapes

:8@*
dtype0
?
Adam/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*,
shared_nameAdam/embedding/embeddings/m
?
/Adam/embedding/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/m*
_output_shapes
:	?@*
dtype0
k

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name113*
value_dtype0	
l
hash_table_1HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name91*
value_dtype0	
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

: *
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
: *
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:@ *
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:@*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	?@*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:?*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
??*
dtype0
?
embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:8@*'
shared_nameembedding_1/embeddings
?
*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings*
_output_shapes

:8@*
dtype0
?
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*%
shared_nameembedding/embeddings
~
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes
:	?@*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R 
?H
Const_2Const*
_output_shapes	
:?*
dtype0*?H
value?HB?H?Bafrica@ara@271Bafrica@eng@124Bafrica@eng@271Bafrica@eng@301Bafrica@eng@537Bafrica@eng@574Bafrica@eng@593Bafrica@eng@601Basia@ara@271Basia@ara@537Basia@ara@601Basia@chi@113Basia@chi@124Basia@chi@183Basia@chi@271Basia@chi@301Basia@chi@445Basia@chi@537Basia@chi@574Basia@chi@588Basia@chi@591Basia@chi@600Basia@chi@601Basia@chi@72Basia@eng@113Basia@eng@124Basia@eng@137Basia@eng@141Basia@eng@144Basia@eng@160Basia@eng@183Basia@eng@229Basia@eng@271Basia@eng@293Basia@eng@301Basia@eng@341Basia@eng@359Basia@eng@388Basia@eng@391Basia@eng@420Basia@eng@438Basia@eng@445Basia@eng@471Basia@eng@490Basia@eng@524Basia@eng@528Basia@eng@537Basia@eng@542Basia@eng@555Basia@eng@557Basia@eng@559Basia@eng@561Basia@eng@562Basia@eng@566Basia@eng@574Basia@eng@58Basia@eng@588Basia@eng@591Basia@eng@592Basia@eng@593Basia@eng@594Basia@eng@600Basia@eng@601Basia@eng@72Basia@gre@537Basia@heb@124Basia@heb@271Basia@heb@341Basia@heb@438Basia@heb@537Basia@heb@574Basia@heb@588Basia@heb@600Basia@heb@601Basia@hin@113Basia@hin@271Basia@hin@601Basia@ind@271Basia@jpn@113Basia@jpn@183Basia@jpn@271Basia@jpn@388Basia@jpn@420Basia@jpn@445Basia@jpn@528Basia@jpn@537Basia@jpn@549Basia@jpn@557Basia@jpn@562Basia@jpn@58Basia@jpn@588Basia@jpn@591Basia@jpn@600Basia@jpn@601Basia@jpn@72Basia@kor@183Basia@kor@271Basia@kor@388Basia@kor@445Basia@kor@460Basia@kor@537Basia@kor@588Basia@kor@601Basia@may@271Basia@may@537Basia@spa@271Basia@tha@271Basia@tha@537Basia@tha@600Basia@tha@601Basia@tur@271Basia@tur@537Basia@tur@574Basia@tur@588Basia@vie@271Basia@vie@524Basia@vie@537Beurope@cze@271Beurope@dan@271Beurope@dan@557Beurope@dan@561Beurope@dut@271Beurope@dut@359Beurope@dut@496Beurope@dut@524Beurope@dut@537Beurope@dut@557Beurope@dut@601Beurope@eng@113Beurope@eng@124Beurope@eng@137Beurope@eng@141Beurope@eng@144Beurope@eng@160Beurope@eng@183Beurope@eng@229Beurope@eng@271Beurope@eng@273Beurope@eng@293Beurope@eng@301Beurope@eng@341Beurope@eng@359Beurope@eng@388Beurope@eng@391Beurope@eng@420Beurope@eng@442Beurope@eng@445Beurope@eng@448Beurope@eng@458Beurope@eng@478Beurope@eng@490Beurope@eng@496Beurope@eng@500Beurope@eng@524Beurope@eng@528Beurope@eng@537Beurope@eng@542Beurope@eng@543Beurope@eng@549Beurope@eng@557Beurope@eng@561Beurope@eng@562Beurope@eng@566Beurope@eng@572Beurope@eng@574Beurope@eng@58Beurope@eng@585Beurope@eng@586Beurope@eng@588Beurope@eng@590Beurope@eng@591Beurope@eng@592Beurope@eng@593Beurope@eng@600Beurope@eng@601Beurope@eng@72Beurope@fin@557Beurope@fin@561Beurope@fre@113Beurope@fre@124Beurope@fre@271Beurope@fre@293Beurope@fre@301Beurope@fre@341Beurope@fre@359Beurope@fre@388Beurope@fre@445Beurope@fre@460Beurope@fre@490Beurope@fre@496Beurope@fre@524Beurope@fre@537Beurope@fre@543Beurope@fre@557Beurope@fre@561Beurope@fre@562Beurope@fre@566Beurope@fre@572Beurope@fre@58Beurope@fre@588Beurope@fre@591Beurope@fre@592Beurope@fre@600Beurope@fre@601Beurope@fre@72Beurope@ger@113Beurope@ger@124Beurope@ger@137Beurope@ger@141Beurope@ger@183Beurope@ger@271Beurope@ger@301Beurope@ger@341Beurope@ger@359Beurope@ger@370Beurope@ger@388Beurope@ger@391Beurope@ger@410Beurope@ger@445Beurope@ger@460Beurope@ger@490Beurope@ger@496Beurope@ger@524Beurope@ger@533Beurope@ger@537Beurope@ger@542Beurope@ger@543Beurope@ger@549Beurope@ger@557Beurope@ger@558Beurope@ger@561Beurope@ger@562Beurope@ger@566Beurope@ger@572Beurope@ger@574Beurope@ger@58Beurope@ger@586Beurope@ger@588Beurope@ger@591Beurope@ger@592Beurope@ger@593Beurope@ger@600Beurope@ger@601Beurope@hun@271Beurope@ita@113Beurope@ita@124Beurope@ita@141Beurope@ita@271Beurope@ita@301Beurope@ita@341Beurope@ita@359Beurope@ita@388Beurope@ita@445Beurope@ita@460Beurope@ita@490Beurope@ita@496Beurope@ita@524Beurope@ita@537Beurope@ita@543Beurope@ita@557Beurope@ita@561Beurope@ita@562Beurope@ita@566Beurope@ita@572Beurope@ita@574Beurope@ita@58Beurope@ita@588Beurope@ita@591Beurope@ita@593Beurope@ita@600Beurope@ita@601Beurope@may@183Beurope@may@301Beurope@may@445Beurope@may@588Beurope@nor@113Beurope@nor@124Beurope@nor@271Beurope@nor@301Beurope@nor@341Beurope@nor@359Beurope@nor@458Beurope@nor@537Beurope@nor@549Beurope@nor@555Beurope@nor@557Beurope@nor@561Beurope@nor@574Beurope@nor@578Beurope@nor@586Beurope@nor@593Beurope@nor@600Beurope@nor@601Beurope@pol@271Beurope@pol@341Beurope@pol@537Beurope@pol@566Beurope@por@271Beurope@por@524Beurope@por@537Beurope@spa@141Beurope@spa@271Beurope@spa@301Beurope@spa@341Beurope@spa@359Beurope@spa@496Beurope@spa@524Beurope@spa@537Beurope@spa@557Beurope@spa@561Beurope@spa@566Beurope@spa@572Beurope@spa@574Beurope@spa@58Beurope@spa@588Beurope@spa@591Beurope@spa@592Beurope@spa@593Beurope@spa@600Beurope@spa@601Beurope@swe@141Beurope@swe@271Beurope@swe@445Beurope@swe@524Beurope@swe@537Beurope@swe@549Beurope@swe@557Beurope@swe@561Beurope@swe@574Beurope@swe@588Beurope@swe@593Beurope@swe@600Beurope@swe@601Bnorth-america@eng@113Bnorth-america@eng@124Bnorth-america@eng@137Bnorth-america@eng@141Bnorth-america@eng@144Bnorth-america@eng@160Bnorth-america@eng@183Bnorth-america@eng@229Bnorth-america@eng@271Bnorth-america@eng@272Bnorth-america@eng@273Bnorth-america@eng@288Bnorth-america@eng@293Bnorth-america@eng@301Bnorth-america@eng@310Bnorth-america@eng@312Bnorth-america@eng@328Bnorth-america@eng@341Bnorth-america@eng@353Bnorth-america@eng@359Bnorth-america@eng@371Bnorth-america@eng@388Bnorth-america@eng@391Bnorth-america@eng@404Bnorth-america@eng@410Bnorth-america@eng@420Bnorth-america@eng@436Bnorth-america@eng@437Bnorth-america@eng@438Bnorth-america@eng@442Bnorth-america@eng@444Bnorth-america@eng@445Bnorth-america@eng@448Bnorth-america@eng@450Bnorth-america@eng@458Bnorth-america@eng@46Bnorth-america@eng@460Bnorth-america@eng@468Bnorth-america@eng@474Bnorth-america@eng@478Bnorth-america@eng@487Bnorth-america@eng@490Bnorth-america@eng@496Bnorth-america@eng@499Bnorth-america@eng@500Bnorth-america@eng@520Bnorth-america@eng@524Bnorth-america@eng@527Bnorth-america@eng@528Bnorth-america@eng@533Bnorth-america@eng@537Bnorth-america@eng@542Bnorth-america@eng@543Bnorth-america@eng@549Bnorth-america@eng@555Bnorth-america@eng@557Bnorth-america@eng@558Bnorth-america@eng@559Bnorth-america@eng@561Bnorth-america@eng@562Bnorth-america@eng@565Bnorth-america@eng@566Bnorth-america@eng@570Bnorth-america@eng@572Bnorth-america@eng@574Bnorth-america@eng@575Bnorth-america@eng@578Bnorth-america@eng@58Bnorth-america@eng@585Bnorth-america@eng@586Bnorth-america@eng@588Bnorth-america@eng@590Bnorth-america@eng@591Bnorth-america@eng@592Bnorth-america@eng@593Bnorth-america@eng@600Bnorth-america@eng@601Bnorth-america@eng@70Bnorth-america@eng@72Bnorth-america@eng@78Bnorth-america@eng@82Bnorth-america@eng@95Bnorth-america@fre@113Bnorth-america@fre@271Bnorth-america@fre@273Bnorth-america@fre@388Bnorth-america@fre@445Bnorth-america@fre@448Bnorth-america@fre@500Bnorth-america@fre@524Bnorth-america@fre@537Bnorth-america@fre@557Bnorth-america@fre@600Bnorth-america@fre@601Bnorth-america@fre@72Bnorth-america@kor@271Bnorth-america@spa@113Bnorth-america@spa@271Bnorth-america@spa@293Bnorth-america@spa@341Bnorth-america@spa@500Bnorth-america@spa@537Bnorth-america@spa@542Bnorth-america@spa@558Bnorth-america@spa@565Bnorth-america@spa@566Bnorth-america@spa@572Bnorth-america@spa@574Bnorth-america@spa@58Bnorth-america@spa@588Bnorth-america@spa@600Bnorth-america@spa@601Boceania@eng@113Boceania@eng@124Boceania@eng@137Boceania@eng@144Boceania@eng@183Boceania@eng@229Boceania@eng@271Boceania@eng@293Boceania@eng@301Boceania@eng@341Boceania@eng@359Boceania@eng@388Boceania@eng@391Boceania@eng@420Boceania@eng@442Boceania@eng@445Boceania@eng@450Boceania@eng@458Boceania@eng@490Boceania@eng@496Boceania@eng@524Boceania@eng@528Boceania@eng@537Boceania@eng@542Boceania@eng@543Boceania@eng@549Boceania@eng@557Boceania@eng@561Boceania@eng@562Boceania@eng@566Boceania@eng@572Boceania@eng@574Boceania@eng@58Boceania@eng@586Boceania@eng@588Boceania@eng@591Boceania@eng@592Boceania@eng@593Boceania@eng@600Boceania@eng@601Boceania@eng@72Bsouth-america@eng@124Bsouth-america@eng@144Bsouth-america@eng@229Bsouth-america@eng@271Bsouth-america@eng@442Bsouth-america@eng@524Bsouth-america@eng@537Bsouth-america@eng@543Bsouth-america@eng@557Bsouth-america@eng@574Bsouth-america@eng@588Bsouth-america@eng@591Bsouth-america@eng@601Bsouth-america@eng@72Bsouth-america@por@113Bsouth-america@por@124Bsouth-america@por@141Bsouth-america@por@160Bsouth-america@por@271Bsouth-america@por@301Bsouth-america@por@341Bsouth-america@por@359Bsouth-america@por@445Bsouth-america@por@490Bsouth-america@por@524Bsouth-america@por@537Bsouth-america@por@557Bsouth-america@por@566Bsouth-america@por@574Bsouth-america@por@588Bsouth-america@por@591Bsouth-america@por@592Bsouth-america@por@593Bsouth-america@por@600Bsouth-america@por@601Bsouth-america@spa@113Bsouth-america@spa@271Bsouth-america@spa@537Bsouth-america@spa@601
?!
Const_3Const*
_output_shapes	
:?*
dtype0	*?!
value? B? 	?"?                                                         	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                        
?
Const_4Const*
_output_shapes
:7*
dtype0*?
value?B?7B	b2b@otherB&consumer_study@finance_legal_insuranceBconsumer_study@travelBconsumer_study@otherBrecruit@single_opt_inBconsumer_study@householdB"consumer_study@shopping_ecommerce Bb2b@banking_financialBb2b@marketing_advertisingB	b2b@salesBb2b@serviceBb2b@hrBb2b@it_decision_makerBb2b@operationsBb2b@technologyBb2b@securityBb2b@software_hardwareBb2b@transportation_logisticsBconsumer_study@entertainmentBhealthcare@patientB!consumer_study@print_social_mediaBconsumer_study@technologyBconsumer_study@tobacco_vaping B%consumer_study@video_games_games_toysB(consumer_study@automotive_transportationB!consumer_study@grooming_cosmeticsBconsumer_study@social_mediaBconsumer_study@sports_gamblingBb2b@office_suppliesBb2b@real_estate_brokers_agentsBconsumer_study@food_beverageBconsumer_study@pets_animalB	b2b@legalBcommunity@otherBcommunity@recruitment_panelB"consumer_study@community_insights Bconsumer_study@news_print_mediaB)consumer_study@nutrition_wellness_fitnessBconsumer_study@politicalB'consumer_study@political_civic_servicesB&consumer_study@realestate_constructionBmusic_study@long_formBmusic_study@shortBb2b@biotech_pharmaceuticalBb2b@corporate_travelBb2b@fulfillmentBhealthcare@otherBhealthcare@professionalBproduct_testing@apparel_fashionB product_testing@beauty_self_careBproduct_testing@electronicsBproduct_testing@pets_animalBrecruit@double_opt_inBrecruit@otherBproduct_testing@other
?
Const_5Const*
_output_shapes
:7*
dtype0	*?
value?B?	7"?                                                        	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       
?
StatefulPartitionedCallStatefulPartitionedCallhash_table_1Const_2Const_3*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *$
fR
__inference_<lambda>_175386
?
StatefulPartitionedCall_1StatefulPartitionedCall
hash_tableConst_4Const_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *$
fR
__inference_<lambda>_175394
B
NoOpNoOp^StatefulPartitionedCall^StatefulPartitionedCall_1
?m
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*?l
value?lB?l B?l
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
ranking_model
	task

	optimizer
loss

signatures*
Z
0
1
2
3
4
5
6
7
8
9
10
11*
Z
0
1
2
3
4
5
6
7
8
9
10
11*
* 
?
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
trace_0
trace_1
 trace_2
!trace_3* 
6
"trace_0
#trace_1
$trace_2
%trace_3* 
* 
?
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,supplier_embeddings
-subject_embeddings
.ratings*
?
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
5_ranking_metrics
6_prediction_metrics
7_label_metrics
8_loss_metrics*
?
9iter

:beta_1

;beta_2
	<decay
=learning_ratem?m?m?m?m?m?m?m?m?m?m?m?v?v?v?v?v?v?v?v?v?v?v?v?*
* 

>serving_default* 
TN
VARIABLE_VALUEembedding/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEembedding_1/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
dense/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_3/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_3/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_4/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_4/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
	1*

?0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Z
0
1
2
3
4
5
6
7
8
9
10
11*
Z
0
1
2
3
4
5
6
7
8
9
10
11*
* 
?
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*
6
Etrace_0
Ftrace_1
Gtrace_2
Htrace_3* 
6
Itrace_0
Jtrace_1
Ktrace_2
Ltrace_3* 
?
Mlayer-0
Nlayer_with_weights-0
Nlayer-1
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses*
?
Ulayer-0
Vlayer_with_weights-0
Vlayer-1
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses*
?
]layer_with_weights-0
]layer-0
^layer_with_weights-1
^layer-1
_layer_with_weights-2
_layer-2
`layer_with_weights-3
`layer-3
alayer_with_weights-4
alayer-4
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses*
* 
* 
* 
?
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*
* 
* 

?0*
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
8
m	variables
n	keras_api
	ototal
	pcount*
* 

,0
-1
.2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
#
q	keras_api
rlookup_table* 
?
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses

embeddings*

0*

0*
* 
?
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*
8
~trace_0
trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
%
?	keras_api
?lookup_table* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

embeddings*

0*

0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias*
J
0
1
2
3
4
5
6
7
8
9*
J
0
1
2
3
4
5
6
7
8
9*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
* 
* 

?0*
* 
!
?root_mean_squared_error*

o0
p1*

m	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
V
?_initializer
?_create_resource
?_initialize
?_destroy_resource* 

0*

0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 

M0
N1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
V
?_initializer
?_create_resource
?_initialize
?_destroy_resource* 

0*

0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 

U0
V1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
'
]0
^1
_2
`3
a4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?trace_0* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 

?trace_0* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
wq
VARIABLE_VALUEAdam/embedding/embeddings/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/embedding_1/embeddings/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEAdam/dense/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_1/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense_1/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_2/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense_2/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_3/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense_3/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_4/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_4/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/embedding/embeddings/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/embedding_1/embeddings/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEAdam/dense/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_1/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense_1/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_2/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense_2/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_3/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense_3/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_4/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_4/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
u
serving_default_subject_idPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_supplier_idPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_2StatefulPartitionedCallserving_default_subject_idserving_default_supplier_idhash_table_1Constembedding/embeddings
hash_tableConst_1embedding_1/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/bias*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_174601
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp*embedding_1/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp/Adam/embedding/embeddings/m/Read/ReadVariableOp1Adam/embedding_1/embeddings/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp/Adam/embedding/embeddings/v/Read/ReadVariableOp1Adam/embedding_1/embeddings/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOpConst_6*8
Tin1
/2-	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_175557
?
StatefulPartitionedCall_4StatefulPartitionedCallsaver_filenameembedding/embeddingsembedding_1/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/embedding/embeddings/mAdam/embedding_1/embeddings/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/embedding/embeddings/vAdam/embedding_1/embeddings/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/v*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_175696??
?
?
P__inference_supplier_recommender_layer_call_and_return_conditional_losses_174406
features

features_1
ranking_model_174372
ranking_model_174374	'
ranking_model_174376:	?@
ranking_model_174378
ranking_model_174380	&
ranking_model_174382:8@(
ranking_model_174384:
??#
ranking_model_174386:	?'
ranking_model_174388:	?@"
ranking_model_174390:@&
ranking_model_174392:@ "
ranking_model_174394: &
ranking_model_174396: "
ranking_model_174398:&
ranking_model_174400:"
ranking_model_174402:
identity??%ranking_model/StatefulPartitionedCall?
%ranking_model/StatefulPartitionedCallStatefulPartitionedCall
features_1featuresranking_model_174372ranking_model_174374ranking_model_174376ranking_model_174378ranking_model_174380ranking_model_174382ranking_model_174384ranking_model_174386ranking_model_174388ranking_model_174390ranking_model_174392ranking_model_174394ranking_model_174396ranking_model_174398ranking_model_174400ranking_model_174402*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_ranking_model_layer_call_and_return_conditional_losses_174092}
IdentityIdentity.ranking_model/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????n
NoOpNoOp&^ranking_model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????:?????????: : : : : : : : : : : : : : : : 2N
%ranking_model/StatefulPartitionedCall%ranking_model/StatefulPartitionedCall:M I
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:

_output_shapes
: :

_output_shapes
: 
?*
?
H__inference_sequential_2_layer_call_and_return_conditional_losses_175211

inputs8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@ 5
'dense_2_biasadd_readvariableop_resource: 8
&dense_3_matmul_readvariableop_resource: 5
'dense_3_biasadd_readvariableop_resource:8
&dense_4_matmul_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0v
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@a
dense_1/LeakyRelu	LeakyReludense_1/BiasAdd:output:0*'
_output_shapes
:?????????@?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0?
dense_2/MatMulMatMuldense_1/LeakyRelu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? a
dense_2/LeakyRelu	LeakyReludense_2/BiasAdd:output:0*'
_output_shapes
:????????? ?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
dense_3/MatMulMatMuldense_2/LeakyRelu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_173558

inputsD
@string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleE
Astring_lookup_1_hash_table_lookup_lookuptablefindv2_default_value	$
embedding_1_173554:8@
identity??#embedding_1/StatefulPartitionedCall?3string_lookup_1/hash_table_Lookup/LookupTableFindV2?
3string_lookup_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2@string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleinputsAstring_lookup_1_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup_1/IdentityIdentity<string_lookup_1/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_1/Identity:output:0embedding_1_173554*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_173512{
IdentityIdentity,embedding_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp$^embedding_1/StatefulPartitionedCall4^string_lookup_1/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2j
3string_lookup_1/hash_table_Lookup/LookupTableFindV23string_lookup_1/hash_table_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?T
?
__inference__traced_save_175557
file_prefix3
/savev2_embedding_embeddings_read_readvariableop5
1savev2_embedding_1_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop:
6savev2_adam_embedding_embeddings_m_read_readvariableop<
8savev2_adam_embedding_1_embeddings_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop:
6savev2_adam_embedding_embeddings_v_read_readvariableop<
8savev2_adam_embedding_1_embeddings_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop
savev2_const_6

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*?
value?B?,B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop1savev2_embedding_1_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop6savev2_adam_embedding_embeddings_m_read_readvariableop8savev2_adam_embedding_1_embeddings_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop6savev2_adam_embedding_embeddings_v_read_readvariableop8savev2_adam_embedding_1_embeddings_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 *:
dtypes0
.2,	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?@:8@:
??:?:	?@:@:@ : : :::: : : : : : : :	?@:8@:
??:?:	?@:@:@ : : ::::	?@:8@:
??:?:	?@:@:@ : : :::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?@:$ 

_output_shapes

:8@:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$	 

_output_shapes

: : 


_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?@:$ 

_output_shapes

:8@:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::% !

_output_shapes
:	?@:$! 

_output_shapes

:8@:&""
 
_output_shapes
:
??:!#

_output_shapes	
:?:%$!

_output_shapes
:	?@: %

_output_shapes
:@:$& 

_output_shapes

:@ : '

_output_shapes
: :$( 

_output_shapes

: : )

_output_shapes
::$* 

_output_shapes

:: +

_output_shapes
::,

_output_shapes
: 
?

?
A__inference_dense_layer_call_and_return_conditional_losses_173618

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_1_layer_call_fn_175272

inputs
unknown:	?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_173635o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference__initializer_1753555
1key_value_init90_lookuptableimportv2_table_handle-
)key_value_init90_lookuptableimportv2_keys/
+key_value_init90_lookuptableimportv2_values	
identity??$key_value_init90/LookupTableImportV2?
$key_value_init90/LookupTableImportV2LookupTableImportV21key_value_init90_lookuptableimportv2_table_handle)key_value_init90_lookuptableimportv2_keys+key_value_init90_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: m
NoOpNoOp%^key_value_init90/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2L
$key_value_init90/LookupTableImportV2$key_value_init90/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?
?
+__inference_sequential_layer_call_fn_175011

inputs
unknown
	unknown_0	
	unknown_1:	?@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_173450o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?V
?
I__inference_ranking_model_layer_call_and_return_conditional_losses_174930
inputs_0
inputs_1M
Isequential_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleN
Jsequential_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	?
,sequential_embedding_embedding_lookup_174879:	?@Q
Msequential_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleR
Nsequential_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_default_value	B
0sequential_1_embedding_1_embedding_lookup_174888:8@E
1sequential_2_dense_matmul_readvariableop_resource:
??A
2sequential_2_dense_biasadd_readvariableop_resource:	?F
3sequential_2_dense_1_matmul_readvariableop_resource:	?@B
4sequential_2_dense_1_biasadd_readvariableop_resource:@E
3sequential_2_dense_2_matmul_readvariableop_resource:@ B
4sequential_2_dense_2_biasadd_readvariableop_resource: E
3sequential_2_dense_3_matmul_readvariableop_resource: B
4sequential_2_dense_3_biasadd_readvariableop_resource:E
3sequential_2_dense_4_matmul_readvariableop_resource:B
4sequential_2_dense_4_biasadd_readvariableop_resource:
identity??%sequential/embedding/embedding_lookup?<sequential/string_lookup/hash_table_Lookup/LookupTableFindV2?)sequential_1/embedding_1/embedding_lookup?@sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2?)sequential_2/dense/BiasAdd/ReadVariableOp?(sequential_2/dense/MatMul/ReadVariableOp?+sequential_2/dense_1/BiasAdd/ReadVariableOp?*sequential_2/dense_1/MatMul/ReadVariableOp?+sequential_2/dense_2/BiasAdd/ReadVariableOp?*sequential_2/dense_2/MatMul/ReadVariableOp?+sequential_2/dense_3/BiasAdd/ReadVariableOp?*sequential_2/dense_3/MatMul/ReadVariableOp?+sequential_2/dense_4/BiasAdd/ReadVariableOp?*sequential_2/dense_4/MatMul/ReadVariableOp?
<sequential/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Isequential_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleinputs_0Jsequential_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
!sequential/string_lookup/IdentityIdentityEsequential/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
%sequential/embedding/embedding_lookupResourceGather,sequential_embedding_embedding_lookup_174879*sequential/string_lookup/Identity:output:0*
Tindices0	*?
_class5
31loc:@sequential/embedding/embedding_lookup/174879*'
_output_shapes
:?????????@*
dtype0?
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0*
T0*?
_class5
31loc:@sequential/embedding/embedding_lookup/174879*'
_output_shapes
:?????????@?
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@?
@sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Msequential_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleinputs_1Nsequential_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
%sequential_1/string_lookup_1/IdentityIdentityIsequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)sequential_1/embedding_1/embedding_lookupResourceGather0sequential_1_embedding_1_embedding_lookup_174888.sequential_1/string_lookup_1/Identity:output:0*
Tindices0	*C
_class9
75loc:@sequential_1/embedding_1/embedding_lookup/174888*'
_output_shapes
:?????????@*
dtype0?
2sequential_1/embedding_1/embedding_lookup/IdentityIdentity2sequential_1/embedding_1/embedding_lookup:output:0*
T0*C
_class9
75loc:@sequential_1/embedding_1/embedding_lookup/174888*'
_output_shapes
:?????????@?
4sequential_1/embedding_1/embedding_lookup/Identity_1Identity;sequential_1/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV29sequential/embedding/embedding_lookup/Identity_1:output:0=sequential_1/embedding_1/embedding_lookup/Identity_1:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
(sequential_2/dense/MatMul/ReadVariableOpReadVariableOp1sequential_2_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential_2/dense/MatMulMatMulconcat:output:00sequential_2/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
)sequential_2/dense/BiasAdd/ReadVariableOpReadVariableOp2sequential_2_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_2/dense/BiasAddBiasAdd#sequential_2/dense/MatMul:product:01sequential_2/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????w
sequential_2/dense/ReluRelu#sequential_2/dense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
*sequential_2/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
sequential_2/dense_1/MatMulMatMul%sequential_2/dense/Relu:activations:02sequential_2/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
+sequential_2/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_2/dense_1/BiasAddBiasAdd%sequential_2/dense_1/MatMul:product:03sequential_2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@{
sequential_2/dense_1/LeakyRelu	LeakyRelu%sequential_2/dense_1/BiasAdd:output:0*'
_output_shapes
:?????????@?
*sequential_2/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0?
sequential_2/dense_2/MatMulMatMul,sequential_2/dense_1/LeakyRelu:activations:02sequential_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
+sequential_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_2/dense_2/BiasAddBiasAdd%sequential_2/dense_2/MatMul:product:03sequential_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? {
sequential_2/dense_2/LeakyRelu	LeakyRelu%sequential_2/dense_2/BiasAdd:output:0*'
_output_shapes
:????????? ?
*sequential_2/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
sequential_2/dense_3/MatMulMatMul,sequential_2/dense_2/LeakyRelu:activations:02sequential_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_2/dense_3/BiasAddBiasAdd%sequential_2/dense_3/MatMul:product:03sequential_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
sequential_2/dense_3/ReluRelu%sequential_2/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
*sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
sequential_2/dense_4/MatMulMatMul'sequential_2/dense_3/Relu:activations:02sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_2/dense_4/BiasAddBiasAdd%sequential_2/dense_4/MatMul:product:03sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
IdentityIdentity%sequential_2/dense_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp&^sequential/embedding/embedding_lookup=^sequential/string_lookup/hash_table_Lookup/LookupTableFindV2*^sequential_1/embedding_1/embedding_lookupA^sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2*^sequential_2/dense/BiasAdd/ReadVariableOp)^sequential_2/dense/MatMul/ReadVariableOp,^sequential_2/dense_1/BiasAdd/ReadVariableOp+^sequential_2/dense_1/MatMul/ReadVariableOp,^sequential_2/dense_2/BiasAdd/ReadVariableOp+^sequential_2/dense_2/MatMul/ReadVariableOp,^sequential_2/dense_3/BiasAdd/ReadVariableOp+^sequential_2/dense_3/MatMul/ReadVariableOp,^sequential_2/dense_4/BiasAdd/ReadVariableOp+^sequential_2/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????:?????????: : : : : : : : : : : : : : : : 2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2|
<sequential/string_lookup/hash_table_Lookup/LookupTableFindV2<sequential/string_lookup/hash_table_Lookup/LookupTableFindV22V
)sequential_1/embedding_1/embedding_lookup)sequential_1/embedding_1/embedding_lookup2?
@sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2@sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV22V
)sequential_2/dense/BiasAdd/ReadVariableOp)sequential_2/dense/BiasAdd/ReadVariableOp2T
(sequential_2/dense/MatMul/ReadVariableOp(sequential_2/dense/MatMul/ReadVariableOp2Z
+sequential_2/dense_1/BiasAdd/ReadVariableOp+sequential_2/dense_1/BiasAdd/ReadVariableOp2X
*sequential_2/dense_1/MatMul/ReadVariableOp*sequential_2/dense_1/MatMul/ReadVariableOp2Z
+sequential_2/dense_2/BiasAdd/ReadVariableOp+sequential_2/dense_2/BiasAdd/ReadVariableOp2X
*sequential_2/dense_2/MatMul/ReadVariableOp*sequential_2/dense_2/MatMul/ReadVariableOp2Z
+sequential_2/dense_3/BiasAdd/ReadVariableOp+sequential_2/dense_3/BiasAdd/ReadVariableOp2X
*sequential_2/dense_3/MatMul/ReadVariableOp*sequential_2/dense_3/MatMul/ReadVariableOp2Z
+sequential_2/dense_4/BiasAdd/ReadVariableOp+sequential_2/dense_4/BiasAdd/ReadVariableOp2X
*sequential_2/dense_4/MatMul/ReadVariableOp*sequential_2/dense_4/MatMul/ReadVariableOp:M I
#
_output_shapes
:?????????
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: 
?

?
-__inference_sequential_2_layer_call_fn_173869
dense_input
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_173821o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:??????????
%
_user_specified_namedense_input
?
?
I__inference_ranking_model_layer_call_and_return_conditional_losses_173974

inputs
inputs_1
sequential_173936
sequential_173938	$
sequential_173940:	?@
sequential_1_173943
sequential_1_173945	%
sequential_1_173947:8@'
sequential_2_173952:
??"
sequential_2_173954:	?&
sequential_2_173956:	?@!
sequential_2_173958:@%
sequential_2_173960:@ !
sequential_2_173962: %
sequential_2_173964: !
sequential_2_173966:%
sequential_2_173968:!
sequential_2_173970:
identity??"sequential/StatefulPartitionedCall?$sequential_1/StatefulPartitionedCall?$sequential_2/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_173936sequential_173938sequential_173940*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_173409?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1sequential_1_173943sequential_1_173945sequential_1_173947*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_173517M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2+sequential/StatefulPartitionedCall:output:0-sequential_1/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0sequential_2_173952sequential_2_173954sequential_2_173956sequential_2_173958sequential_2_173960sequential_2_173962sequential_2_173964sequential_2_173966sequential_2_173968sequential_2_173970*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_173692|
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????:?????????: : : : : : : : : : : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
H__inference_sequential_2_layer_call_and_return_conditional_losses_173821

inputs 
dense_173795:
??
dense_173797:	?!
dense_1_173800:	?@
dense_1_173802:@ 
dense_2_173805:@ 
dense_2_173807:  
dense_3_173810: 
dense_3_173812: 
dense_4_173815:
dense_4_173817:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_173795dense_173797*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_173618?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_173800dense_1_173802*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_173635?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_173805dense_2_173807*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_173652?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_173810dense_3_173812*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_173669?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_173815dense_4_173817*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_173685w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?*
?
H__inference_sequential_2_layer_call_and_return_conditional_losses_175173

inputs8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@ 5
'dense_2_biasadd_readvariableop_resource: 8
&dense_3_matmul_readvariableop_resource: 5
'dense_3_biasadd_readvariableop_resource:8
&dense_4_matmul_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0v
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@a
dense_1/LeakyRelu	LeakyReludense_1/BiasAdd:output:0*'
_output_shapes
:?????????@?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0?
dense_2/MatMulMatMuldense_1/LeakyRelu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? a
dense_2/LeakyRelu	LeakyReludense_2/BiasAdd:output:0*'
_output_shapes
:????????? ?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
dense_3/MatMulMatMuldense_2/LeakyRelu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_ranking_model_layer_call_fn_174871
inputs_0
inputs_1
unknown
	unknown_0	
	unknown_1:	?@
	unknown_2
	unknown_3	
	unknown_4:8@
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?@
	unknown_8:@
	unknown_9:@ 

unknown_10: 

unknown_11: 

unknown_12:

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_ranking_model_layer_call_and_return_conditional_losses_174092o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:?????????
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_sequential_layer_call_fn_173418
string_lookup_input
unknown
	unknown_0	
	unknown_1:	?@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_173409o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
#
_output_shapes
:?????????
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
?
?
+__inference_sequential_layer_call_fn_173470
string_lookup_input
unknown
	unknown_0	
	unknown_1:	?@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_173450o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
#
_output_shapes
:?????????
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
?
?
.__inference_ranking_model_layer_call_fn_174009
input_1
input_2
unknown
	unknown_0	
	unknown_1:	?@
	unknown_2
	unknown_3	
	unknown_4:8@
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?@
	unknown_8:@
	unknown_9:@ 

unknown_10: 

unknown_11: 

unknown_12:

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_ranking_model_layer_call_and_return_conditional_losses_173974o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:LH
#
_output_shapes
:?????????
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: 
?

?
C__inference_dense_3_layer_call_and_return_conditional_losses_175323

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_174601

subject_id
supplier_id
unknown
	unknown_0	
	unknown_1:	?@
	unknown_2
	unknown_3	
	unknown_4:8@
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?@
	unknown_8:@
	unknown_9:@ 

unknown_10: 

unknown_11: 

unknown_12:

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
subject_idsupplier_idunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_173384o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:?????????
$
_user_specified_name
subject_id:PL
#
_output_shapes
:?????????
%
_user_specified_namesupplier_id:

_output_shapes
: :

_output_shapes
: 
?
?
5__inference_supplier_recommender_layer_call_fn_174479

subject_id
supplier_id
unknown
	unknown_0	
	unknown_1:	?@
	unknown_2
	unknown_3	
	unknown_4:8@
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?@
	unknown_8:@
	unknown_9:@ 

unknown_10: 

unknown_11: 

unknown_12:

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
subject_idsupplier_idunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_supplier_recommender_layer_call_and_return_conditional_losses_174406o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:?????????
$
_user_specified_name
subject_id:PL
#
_output_shapes
:?????????
%
_user_specified_namesupplier_id:

_output_shapes
: :

_output_shapes
: 
?

?
C__inference_dense_2_layer_call_and_return_conditional_losses_175303

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? Q
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:????????? f
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
C__inference_dense_2_layer_call_and_return_conditional_losses_173652

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? Q
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:????????? f
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_175037

inputsB
>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleC
?string_lookup_hash_table_lookup_lookuptablefindv2_default_value	4
!embedding_embedding_lookup_175031:	?@
identity??embedding/embedding_lookup?1string_lookup/hash_table_Lookup/LookupTableFindV2?
1string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleinputs?string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup/IdentityIdentity:string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_175031string_lookup/Identity:output:0*
Tindices0	*4
_class*
(&loc:@embedding/embedding_lookup/175031*'
_output_shapes
:?????????@*
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/175031*'
_output_shapes
:?????????@?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@}
IdentityIdentity.embedding/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^embedding/embedding_lookup2^string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 28
embedding/embedding_lookupembedding/embedding_lookup2f
1string_lookup/hash_table_Lookup/LookupTableFindV21string_lookup/hash_table_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?h
?
P__inference_supplier_recommender_layer_call_and_return_conditional_losses_174736
features_subject_id
features_supplier_id[
Wranking_model_sequential_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle\
Xranking_model_sequential_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	M
:ranking_model_sequential_embedding_embedding_lookup_174685:	?@_
[ranking_model_sequential_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handle`
\ranking_model_sequential_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_default_value	P
>ranking_model_sequential_1_embedding_1_embedding_lookup_174694:8@S
?ranking_model_sequential_2_dense_matmul_readvariableop_resource:
??O
@ranking_model_sequential_2_dense_biasadd_readvariableop_resource:	?T
Aranking_model_sequential_2_dense_1_matmul_readvariableop_resource:	?@P
Branking_model_sequential_2_dense_1_biasadd_readvariableop_resource:@S
Aranking_model_sequential_2_dense_2_matmul_readvariableop_resource:@ P
Branking_model_sequential_2_dense_2_biasadd_readvariableop_resource: S
Aranking_model_sequential_2_dense_3_matmul_readvariableop_resource: P
Branking_model_sequential_2_dense_3_biasadd_readvariableop_resource:S
Aranking_model_sequential_2_dense_4_matmul_readvariableop_resource:P
Branking_model_sequential_2_dense_4_biasadd_readvariableop_resource:
identity??3ranking_model/sequential/embedding/embedding_lookup?Jranking_model/sequential/string_lookup/hash_table_Lookup/LookupTableFindV2?7ranking_model/sequential_1/embedding_1/embedding_lookup?Nranking_model/sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2?7ranking_model/sequential_2/dense/BiasAdd/ReadVariableOp?6ranking_model/sequential_2/dense/MatMul/ReadVariableOp?9ranking_model/sequential_2/dense_1/BiasAdd/ReadVariableOp?8ranking_model/sequential_2/dense_1/MatMul/ReadVariableOp?9ranking_model/sequential_2/dense_2/BiasAdd/ReadVariableOp?8ranking_model/sequential_2/dense_2/MatMul/ReadVariableOp?9ranking_model/sequential_2/dense_3/BiasAdd/ReadVariableOp?8ranking_model/sequential_2/dense_3/MatMul/ReadVariableOp?9ranking_model/sequential_2/dense_4/BiasAdd/ReadVariableOp?8ranking_model/sequential_2/dense_4/MatMul/ReadVariableOp?
Jranking_model/sequential/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Wranking_model_sequential_string_lookup_hash_table_lookup_lookuptablefindv2_table_handlefeatures_supplier_idXranking_model_sequential_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
/ranking_model/sequential/string_lookup/IdentityIdentitySranking_model/sequential/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
3ranking_model/sequential/embedding/embedding_lookupResourceGather:ranking_model_sequential_embedding_embedding_lookup_1746858ranking_model/sequential/string_lookup/Identity:output:0*
Tindices0	*M
_classC
A?loc:@ranking_model/sequential/embedding/embedding_lookup/174685*'
_output_shapes
:?????????@*
dtype0?
<ranking_model/sequential/embedding/embedding_lookup/IdentityIdentity<ranking_model/sequential/embedding/embedding_lookup:output:0*
T0*M
_classC
A?loc:@ranking_model/sequential/embedding/embedding_lookup/174685*'
_output_shapes
:?????????@?
>ranking_model/sequential/embedding/embedding_lookup/Identity_1IdentityEranking_model/sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@?
Nranking_model/sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2[ranking_model_sequential_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handlefeatures_subject_id\ranking_model_sequential_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
3ranking_model/sequential_1/string_lookup_1/IdentityIdentityWranking_model/sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
7ranking_model/sequential_1/embedding_1/embedding_lookupResourceGather>ranking_model_sequential_1_embedding_1_embedding_lookup_174694<ranking_model/sequential_1/string_lookup_1/Identity:output:0*
Tindices0	*Q
_classG
ECloc:@ranking_model/sequential_1/embedding_1/embedding_lookup/174694*'
_output_shapes
:?????????@*
dtype0?
@ranking_model/sequential_1/embedding_1/embedding_lookup/IdentityIdentity@ranking_model/sequential_1/embedding_1/embedding_lookup:output:0*
T0*Q
_classG
ECloc:@ranking_model/sequential_1/embedding_1/embedding_lookup/174694*'
_output_shapes
:?????????@?
Branking_model/sequential_1/embedding_1/embedding_lookup/Identity_1IdentityIranking_model/sequential_1/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@[
ranking_model/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
ranking_model/concatConcatV2Granking_model/sequential/embedding/embedding_lookup/Identity_1:output:0Kranking_model/sequential_1/embedding_1/embedding_lookup/Identity_1:output:0"ranking_model/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
6ranking_model/sequential_2/dense/MatMul/ReadVariableOpReadVariableOp?ranking_model_sequential_2_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
'ranking_model/sequential_2/dense/MatMulMatMulranking_model/concat:output:0>ranking_model/sequential_2/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
7ranking_model/sequential_2/dense/BiasAdd/ReadVariableOpReadVariableOp@ranking_model_sequential_2_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
(ranking_model/sequential_2/dense/BiasAddBiasAdd1ranking_model/sequential_2/dense/MatMul:product:0?ranking_model/sequential_2/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
%ranking_model/sequential_2/dense/ReluRelu1ranking_model/sequential_2/dense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
8ranking_model/sequential_2/dense_1/MatMul/ReadVariableOpReadVariableOpAranking_model_sequential_2_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
)ranking_model/sequential_2/dense_1/MatMulMatMul3ranking_model/sequential_2/dense/Relu:activations:0@ranking_model/sequential_2/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
9ranking_model/sequential_2/dense_1/BiasAdd/ReadVariableOpReadVariableOpBranking_model_sequential_2_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
*ranking_model/sequential_2/dense_1/BiasAddBiasAdd3ranking_model/sequential_2/dense_1/MatMul:product:0Aranking_model/sequential_2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
,ranking_model/sequential_2/dense_1/LeakyRelu	LeakyRelu3ranking_model/sequential_2/dense_1/BiasAdd:output:0*'
_output_shapes
:?????????@?
8ranking_model/sequential_2/dense_2/MatMul/ReadVariableOpReadVariableOpAranking_model_sequential_2_dense_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0?
)ranking_model/sequential_2/dense_2/MatMulMatMul:ranking_model/sequential_2/dense_1/LeakyRelu:activations:0@ranking_model/sequential_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
9ranking_model/sequential_2/dense_2/BiasAdd/ReadVariableOpReadVariableOpBranking_model_sequential_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
*ranking_model/sequential_2/dense_2/BiasAddBiasAdd3ranking_model/sequential_2/dense_2/MatMul:product:0Aranking_model/sequential_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
,ranking_model/sequential_2/dense_2/LeakyRelu	LeakyRelu3ranking_model/sequential_2/dense_2/BiasAdd:output:0*'
_output_shapes
:????????? ?
8ranking_model/sequential_2/dense_3/MatMul/ReadVariableOpReadVariableOpAranking_model_sequential_2_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
)ranking_model/sequential_2/dense_3/MatMulMatMul:ranking_model/sequential_2/dense_2/LeakyRelu:activations:0@ranking_model/sequential_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
9ranking_model/sequential_2/dense_3/BiasAdd/ReadVariableOpReadVariableOpBranking_model_sequential_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
*ranking_model/sequential_2/dense_3/BiasAddBiasAdd3ranking_model/sequential_2/dense_3/MatMul:product:0Aranking_model/sequential_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
'ranking_model/sequential_2/dense_3/ReluRelu3ranking_model/sequential_2/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
8ranking_model/sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOpAranking_model_sequential_2_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
)ranking_model/sequential_2/dense_4/MatMulMatMul5ranking_model/sequential_2/dense_3/Relu:activations:0@ranking_model/sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
9ranking_model/sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOpBranking_model_sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
*ranking_model/sequential_2/dense_4/BiasAddBiasAdd3ranking_model/sequential_2/dense_4/MatMul:product:0Aranking_model/sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
IdentityIdentity3ranking_model/sequential_2/dense_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp4^ranking_model/sequential/embedding/embedding_lookupK^ranking_model/sequential/string_lookup/hash_table_Lookup/LookupTableFindV28^ranking_model/sequential_1/embedding_1/embedding_lookupO^ranking_model/sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV28^ranking_model/sequential_2/dense/BiasAdd/ReadVariableOp7^ranking_model/sequential_2/dense/MatMul/ReadVariableOp:^ranking_model/sequential_2/dense_1/BiasAdd/ReadVariableOp9^ranking_model/sequential_2/dense_1/MatMul/ReadVariableOp:^ranking_model/sequential_2/dense_2/BiasAdd/ReadVariableOp9^ranking_model/sequential_2/dense_2/MatMul/ReadVariableOp:^ranking_model/sequential_2/dense_3/BiasAdd/ReadVariableOp9^ranking_model/sequential_2/dense_3/MatMul/ReadVariableOp:^ranking_model/sequential_2/dense_4/BiasAdd/ReadVariableOp9^ranking_model/sequential_2/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????:?????????: : : : : : : : : : : : : : : : 2j
3ranking_model/sequential/embedding/embedding_lookup3ranking_model/sequential/embedding/embedding_lookup2?
Jranking_model/sequential/string_lookup/hash_table_Lookup/LookupTableFindV2Jranking_model/sequential/string_lookup/hash_table_Lookup/LookupTableFindV22r
7ranking_model/sequential_1/embedding_1/embedding_lookup7ranking_model/sequential_1/embedding_1/embedding_lookup2?
Nranking_model/sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2Nranking_model/sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV22r
7ranking_model/sequential_2/dense/BiasAdd/ReadVariableOp7ranking_model/sequential_2/dense/BiasAdd/ReadVariableOp2p
6ranking_model/sequential_2/dense/MatMul/ReadVariableOp6ranking_model/sequential_2/dense/MatMul/ReadVariableOp2v
9ranking_model/sequential_2/dense_1/BiasAdd/ReadVariableOp9ranking_model/sequential_2/dense_1/BiasAdd/ReadVariableOp2t
8ranking_model/sequential_2/dense_1/MatMul/ReadVariableOp8ranking_model/sequential_2/dense_1/MatMul/ReadVariableOp2v
9ranking_model/sequential_2/dense_2/BiasAdd/ReadVariableOp9ranking_model/sequential_2/dense_2/BiasAdd/ReadVariableOp2t
8ranking_model/sequential_2/dense_2/MatMul/ReadVariableOp8ranking_model/sequential_2/dense_2/MatMul/ReadVariableOp2v
9ranking_model/sequential_2/dense_3/BiasAdd/ReadVariableOp9ranking_model/sequential_2/dense_3/BiasAdd/ReadVariableOp2t
8ranking_model/sequential_2/dense_3/MatMul/ReadVariableOp8ranking_model/sequential_2/dense_3/MatMul/ReadVariableOp2v
9ranking_model/sequential_2/dense_4/BiasAdd/ReadVariableOp9ranking_model/sequential_2/dense_4/BiasAdd/ReadVariableOp2t
8ranking_model/sequential_2/dense_4/MatMul/ReadVariableOp8ranking_model/sequential_2/dense_4/MatMul/ReadVariableOp:X T
#
_output_shapes
:?????????
-
_user_specified_namefeatures/subject_id:YU
#
_output_shapes
:?????????
.
_user_specified_namefeatures/supplier_id:

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_dense_3_layer_call_fn_175312

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_173669o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_173409

inputsB
>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleC
?string_lookup_hash_table_lookup_lookuptablefindv2_default_value	#
embedding_173405:	?@
identity??!embedding/StatefulPartitionedCall?1string_lookup/hash_table_Lookup/LookupTableFindV2?
1string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleinputs?string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup/IdentityIdentity:string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
!embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_173405*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_173404y
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp"^embedding/StatefulPartitionedCall2^string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2f
1string_lookup/hash_table_Lookup/LookupTableFindV21string_lookup/hash_table_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
5__inference_supplier_recommender_layer_call_fn_174639
features_subject_id
features_supplier_id
unknown
	unknown_0	
	unknown_1:	?@
	unknown_2
	unknown_3	
	unknown_4:8@
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?@
	unknown_8:@
	unknown_9:@ 

unknown_10: 

unknown_11: 

unknown_12:

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallfeatures_subject_idfeatures_supplier_idunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_supplier_recommender_layer_call_and_return_conditional_losses_174292o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
#
_output_shapes
:?????????
-
_user_specified_namefeatures/subject_id:YU
#
_output_shapes
:?????????
.
_user_specified_namefeatures/supplier_id:

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_173492
string_lookup_inputB
>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleC
?string_lookup_hash_table_lookup_lookuptablefindv2_default_value	#
embedding_173488:	?@
identity??!embedding/StatefulPartitionedCall?1string_lookup/hash_table_Lookup/LookupTableFindV2?
1string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2>string_lookup_hash_table_lookup_lookuptablefindv2_table_handlestring_lookup_input?string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup/IdentityIdentity:string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
!embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_173488*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_173404y
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp"^embedding/StatefulPartitionedCall2^string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2f
1string_lookup/hash_table_Lookup/LookupTableFindV21string_lookup/hash_table_Lookup/LookupTableFindV2:X T
#
_output_shapes
:?????????
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
?
?
G__inference_embedding_1_layer_call_and_return_conditional_losses_175243

inputs	)
embedding_lookup_175237:8@
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_175237inputs*
Tindices0	**
_class 
loc:@embedding_lookup/175237*'
_output_shapes
:?????????@*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/175237*'
_output_shapes
:?????????@}
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????@Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_173450

inputsB
>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleC
?string_lookup_hash_table_lookup_lookuptablefindv2_default_value	#
embedding_173446:	?@
identity??!embedding/StatefulPartitionedCall?1string_lookup/hash_table_Lookup/LookupTableFindV2?
1string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleinputs?string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup/IdentityIdentity:string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
!embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_173446*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_173404y
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp"^embedding/StatefulPartitionedCall2^string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2f
1string_lookup/hash_table_Lookup/LookupTableFindV21string_lookup/hash_table_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
??
?
!__inference__wrapped_model_173384

subject_id
supplier_idp
lsupplier_recommender_ranking_model_sequential_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleq
msupplier_recommender_ranking_model_sequential_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	b
Osupplier_recommender_ranking_model_sequential_embedding_embedding_lookup_173333:	?@t
psupplier_recommender_ranking_model_sequential_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleu
qsupplier_recommender_ranking_model_sequential_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_default_value	e
Ssupplier_recommender_ranking_model_sequential_1_embedding_1_embedding_lookup_173342:8@h
Tsupplier_recommender_ranking_model_sequential_2_dense_matmul_readvariableop_resource:
??d
Usupplier_recommender_ranking_model_sequential_2_dense_biasadd_readvariableop_resource:	?i
Vsupplier_recommender_ranking_model_sequential_2_dense_1_matmul_readvariableop_resource:	?@e
Wsupplier_recommender_ranking_model_sequential_2_dense_1_biasadd_readvariableop_resource:@h
Vsupplier_recommender_ranking_model_sequential_2_dense_2_matmul_readvariableop_resource:@ e
Wsupplier_recommender_ranking_model_sequential_2_dense_2_biasadd_readvariableop_resource: h
Vsupplier_recommender_ranking_model_sequential_2_dense_3_matmul_readvariableop_resource: e
Wsupplier_recommender_ranking_model_sequential_2_dense_3_biasadd_readvariableop_resource:h
Vsupplier_recommender_ranking_model_sequential_2_dense_4_matmul_readvariableop_resource:e
Wsupplier_recommender_ranking_model_sequential_2_dense_4_biasadd_readvariableop_resource:
identity??Hsupplier_recommender/ranking_model/sequential/embedding/embedding_lookup?_supplier_recommender/ranking_model/sequential/string_lookup/hash_table_Lookup/LookupTableFindV2?Lsupplier_recommender/ranking_model/sequential_1/embedding_1/embedding_lookup?csupplier_recommender/ranking_model/sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2?Lsupplier_recommender/ranking_model/sequential_2/dense/BiasAdd/ReadVariableOp?Ksupplier_recommender/ranking_model/sequential_2/dense/MatMul/ReadVariableOp?Nsupplier_recommender/ranking_model/sequential_2/dense_1/BiasAdd/ReadVariableOp?Msupplier_recommender/ranking_model/sequential_2/dense_1/MatMul/ReadVariableOp?Nsupplier_recommender/ranking_model/sequential_2/dense_2/BiasAdd/ReadVariableOp?Msupplier_recommender/ranking_model/sequential_2/dense_2/MatMul/ReadVariableOp?Nsupplier_recommender/ranking_model/sequential_2/dense_3/BiasAdd/ReadVariableOp?Msupplier_recommender/ranking_model/sequential_2/dense_3/MatMul/ReadVariableOp?Nsupplier_recommender/ranking_model/sequential_2/dense_4/BiasAdd/ReadVariableOp?Msupplier_recommender/ranking_model/sequential_2/dense_4/MatMul/ReadVariableOp?
_supplier_recommender/ranking_model/sequential/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2lsupplier_recommender_ranking_model_sequential_string_lookup_hash_table_lookup_lookuptablefindv2_table_handlesupplier_idmsupplier_recommender_ranking_model_sequential_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Dsupplier_recommender/ranking_model/sequential/string_lookup/IdentityIdentityhsupplier_recommender/ranking_model/sequential/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
Hsupplier_recommender/ranking_model/sequential/embedding/embedding_lookupResourceGatherOsupplier_recommender_ranking_model_sequential_embedding_embedding_lookup_173333Msupplier_recommender/ranking_model/sequential/string_lookup/Identity:output:0*
Tindices0	*b
_classX
VTloc:@supplier_recommender/ranking_model/sequential/embedding/embedding_lookup/173333*'
_output_shapes
:?????????@*
dtype0?
Qsupplier_recommender/ranking_model/sequential/embedding/embedding_lookup/IdentityIdentityQsupplier_recommender/ranking_model/sequential/embedding/embedding_lookup:output:0*
T0*b
_classX
VTloc:@supplier_recommender/ranking_model/sequential/embedding/embedding_lookup/173333*'
_output_shapes
:?????????@?
Ssupplier_recommender/ranking_model/sequential/embedding/embedding_lookup/Identity_1IdentityZsupplier_recommender/ranking_model/sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@?
csupplier_recommender/ranking_model/sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2psupplier_recommender_ranking_model_sequential_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handle
subject_idqsupplier_recommender_ranking_model_sequential_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Hsupplier_recommender/ranking_model/sequential_1/string_lookup_1/IdentityIdentitylsupplier_recommender/ranking_model/sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
Lsupplier_recommender/ranking_model/sequential_1/embedding_1/embedding_lookupResourceGatherSsupplier_recommender_ranking_model_sequential_1_embedding_1_embedding_lookup_173342Qsupplier_recommender/ranking_model/sequential_1/string_lookup_1/Identity:output:0*
Tindices0	*f
_class\
ZXloc:@supplier_recommender/ranking_model/sequential_1/embedding_1/embedding_lookup/173342*'
_output_shapes
:?????????@*
dtype0?
Usupplier_recommender/ranking_model/sequential_1/embedding_1/embedding_lookup/IdentityIdentityUsupplier_recommender/ranking_model/sequential_1/embedding_1/embedding_lookup:output:0*
T0*f
_class\
ZXloc:@supplier_recommender/ranking_model/sequential_1/embedding_1/embedding_lookup/173342*'
_output_shapes
:?????????@?
Wsupplier_recommender/ranking_model/sequential_1/embedding_1/embedding_lookup/Identity_1Identity^supplier_recommender/ranking_model/sequential_1/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@p
.supplier_recommender/ranking_model/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
)supplier_recommender/ranking_model/concatConcatV2\supplier_recommender/ranking_model/sequential/embedding/embedding_lookup/Identity_1:output:0`supplier_recommender/ranking_model/sequential_1/embedding_1/embedding_lookup/Identity_1:output:07supplier_recommender/ranking_model/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
Ksupplier_recommender/ranking_model/sequential_2/dense/MatMul/ReadVariableOpReadVariableOpTsupplier_recommender_ranking_model_sequential_2_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
<supplier_recommender/ranking_model/sequential_2/dense/MatMulMatMul2supplier_recommender/ranking_model/concat:output:0Ssupplier_recommender/ranking_model/sequential_2/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
Lsupplier_recommender/ranking_model/sequential_2/dense/BiasAdd/ReadVariableOpReadVariableOpUsupplier_recommender_ranking_model_sequential_2_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
=supplier_recommender/ranking_model/sequential_2/dense/BiasAddBiasAddFsupplier_recommender/ranking_model/sequential_2/dense/MatMul:product:0Tsupplier_recommender/ranking_model/sequential_2/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
:supplier_recommender/ranking_model/sequential_2/dense/ReluReluFsupplier_recommender/ranking_model/sequential_2/dense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
Msupplier_recommender/ranking_model/sequential_2/dense_1/MatMul/ReadVariableOpReadVariableOpVsupplier_recommender_ranking_model_sequential_2_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
>supplier_recommender/ranking_model/sequential_2/dense_1/MatMulMatMulHsupplier_recommender/ranking_model/sequential_2/dense/Relu:activations:0Usupplier_recommender/ranking_model/sequential_2/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Nsupplier_recommender/ranking_model/sequential_2/dense_1/BiasAdd/ReadVariableOpReadVariableOpWsupplier_recommender_ranking_model_sequential_2_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
?supplier_recommender/ranking_model/sequential_2/dense_1/BiasAddBiasAddHsupplier_recommender/ranking_model/sequential_2/dense_1/MatMul:product:0Vsupplier_recommender/ranking_model/sequential_2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Asupplier_recommender/ranking_model/sequential_2/dense_1/LeakyRelu	LeakyReluHsupplier_recommender/ranking_model/sequential_2/dense_1/BiasAdd:output:0*'
_output_shapes
:?????????@?
Msupplier_recommender/ranking_model/sequential_2/dense_2/MatMul/ReadVariableOpReadVariableOpVsupplier_recommender_ranking_model_sequential_2_dense_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0?
>supplier_recommender/ranking_model/sequential_2/dense_2/MatMulMatMulOsupplier_recommender/ranking_model/sequential_2/dense_1/LeakyRelu:activations:0Usupplier_recommender/ranking_model/sequential_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
Nsupplier_recommender/ranking_model/sequential_2/dense_2/BiasAdd/ReadVariableOpReadVariableOpWsupplier_recommender_ranking_model_sequential_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
?supplier_recommender/ranking_model/sequential_2/dense_2/BiasAddBiasAddHsupplier_recommender/ranking_model/sequential_2/dense_2/MatMul:product:0Vsupplier_recommender/ranking_model/sequential_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
Asupplier_recommender/ranking_model/sequential_2/dense_2/LeakyRelu	LeakyReluHsupplier_recommender/ranking_model/sequential_2/dense_2/BiasAdd:output:0*'
_output_shapes
:????????? ?
Msupplier_recommender/ranking_model/sequential_2/dense_3/MatMul/ReadVariableOpReadVariableOpVsupplier_recommender_ranking_model_sequential_2_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
>supplier_recommender/ranking_model/sequential_2/dense_3/MatMulMatMulOsupplier_recommender/ranking_model/sequential_2/dense_2/LeakyRelu:activations:0Usupplier_recommender/ranking_model/sequential_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Nsupplier_recommender/ranking_model/sequential_2/dense_3/BiasAdd/ReadVariableOpReadVariableOpWsupplier_recommender_ranking_model_sequential_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
?supplier_recommender/ranking_model/sequential_2/dense_3/BiasAddBiasAddHsupplier_recommender/ranking_model/sequential_2/dense_3/MatMul:product:0Vsupplier_recommender/ranking_model/sequential_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
<supplier_recommender/ranking_model/sequential_2/dense_3/ReluReluHsupplier_recommender/ranking_model/sequential_2/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
Msupplier_recommender/ranking_model/sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOpVsupplier_recommender_ranking_model_sequential_2_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
>supplier_recommender/ranking_model/sequential_2/dense_4/MatMulMatMulJsupplier_recommender/ranking_model/sequential_2/dense_3/Relu:activations:0Usupplier_recommender/ranking_model/sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Nsupplier_recommender/ranking_model/sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOpWsupplier_recommender_ranking_model_sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
?supplier_recommender/ranking_model/sequential_2/dense_4/BiasAddBiasAddHsupplier_recommender/ranking_model/sequential_2/dense_4/MatMul:product:0Vsupplier_recommender/ranking_model/sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
IdentityIdentityHsupplier_recommender/ranking_model/sequential_2/dense_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????	
NoOpNoOpI^supplier_recommender/ranking_model/sequential/embedding/embedding_lookup`^supplier_recommender/ranking_model/sequential/string_lookup/hash_table_Lookup/LookupTableFindV2M^supplier_recommender/ranking_model/sequential_1/embedding_1/embedding_lookupd^supplier_recommender/ranking_model/sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2M^supplier_recommender/ranking_model/sequential_2/dense/BiasAdd/ReadVariableOpL^supplier_recommender/ranking_model/sequential_2/dense/MatMul/ReadVariableOpO^supplier_recommender/ranking_model/sequential_2/dense_1/BiasAdd/ReadVariableOpN^supplier_recommender/ranking_model/sequential_2/dense_1/MatMul/ReadVariableOpO^supplier_recommender/ranking_model/sequential_2/dense_2/BiasAdd/ReadVariableOpN^supplier_recommender/ranking_model/sequential_2/dense_2/MatMul/ReadVariableOpO^supplier_recommender/ranking_model/sequential_2/dense_3/BiasAdd/ReadVariableOpN^supplier_recommender/ranking_model/sequential_2/dense_3/MatMul/ReadVariableOpO^supplier_recommender/ranking_model/sequential_2/dense_4/BiasAdd/ReadVariableOpN^supplier_recommender/ranking_model/sequential_2/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????:?????????: : : : : : : : : : : : : : : : 2?
Hsupplier_recommender/ranking_model/sequential/embedding/embedding_lookupHsupplier_recommender/ranking_model/sequential/embedding/embedding_lookup2?
_supplier_recommender/ranking_model/sequential/string_lookup/hash_table_Lookup/LookupTableFindV2_supplier_recommender/ranking_model/sequential/string_lookup/hash_table_Lookup/LookupTableFindV22?
Lsupplier_recommender/ranking_model/sequential_1/embedding_1/embedding_lookupLsupplier_recommender/ranking_model/sequential_1/embedding_1/embedding_lookup2?
csupplier_recommender/ranking_model/sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2csupplier_recommender/ranking_model/sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV22?
Lsupplier_recommender/ranking_model/sequential_2/dense/BiasAdd/ReadVariableOpLsupplier_recommender/ranking_model/sequential_2/dense/BiasAdd/ReadVariableOp2?
Ksupplier_recommender/ranking_model/sequential_2/dense/MatMul/ReadVariableOpKsupplier_recommender/ranking_model/sequential_2/dense/MatMul/ReadVariableOp2?
Nsupplier_recommender/ranking_model/sequential_2/dense_1/BiasAdd/ReadVariableOpNsupplier_recommender/ranking_model/sequential_2/dense_1/BiasAdd/ReadVariableOp2?
Msupplier_recommender/ranking_model/sequential_2/dense_1/MatMul/ReadVariableOpMsupplier_recommender/ranking_model/sequential_2/dense_1/MatMul/ReadVariableOp2?
Nsupplier_recommender/ranking_model/sequential_2/dense_2/BiasAdd/ReadVariableOpNsupplier_recommender/ranking_model/sequential_2/dense_2/BiasAdd/ReadVariableOp2?
Msupplier_recommender/ranking_model/sequential_2/dense_2/MatMul/ReadVariableOpMsupplier_recommender/ranking_model/sequential_2/dense_2/MatMul/ReadVariableOp2?
Nsupplier_recommender/ranking_model/sequential_2/dense_3/BiasAdd/ReadVariableOpNsupplier_recommender/ranking_model/sequential_2/dense_3/BiasAdd/ReadVariableOp2?
Msupplier_recommender/ranking_model/sequential_2/dense_3/MatMul/ReadVariableOpMsupplier_recommender/ranking_model/sequential_2/dense_3/MatMul/ReadVariableOp2?
Nsupplier_recommender/ranking_model/sequential_2/dense_4/BiasAdd/ReadVariableOpNsupplier_recommender/ranking_model/sequential_2/dense_4/BiasAdd/ReadVariableOp2?
Msupplier_recommender/ranking_model/sequential_2/dense_4/MatMul/ReadVariableOpMsupplier_recommender/ranking_model/sequential_2/dense_4/MatMul/ReadVariableOp:O K
#
_output_shapes
:?????????
$
_user_specified_name
subject_id:PL
#
_output_shapes
:?????????
%
_user_specified_namesupplier_id:

_output_shapes
: :

_output_shapes
: 
?
?
P__inference_supplier_recommender_layer_call_and_return_conditional_losses_174292
features

features_1
ranking_model_174258
ranking_model_174260	'
ranking_model_174262:	?@
ranking_model_174264
ranking_model_174266	&
ranking_model_174268:8@(
ranking_model_174270:
??#
ranking_model_174272:	?'
ranking_model_174274:	?@"
ranking_model_174276:@&
ranking_model_174278:@ "
ranking_model_174280: &
ranking_model_174282: "
ranking_model_174284:&
ranking_model_174286:"
ranking_model_174288:
identity??%ranking_model/StatefulPartitionedCall?
%ranking_model/StatefulPartitionedCallStatefulPartitionedCall
features_1featuresranking_model_174258ranking_model_174260ranking_model_174262ranking_model_174264ranking_model_174266ranking_model_174268ranking_model_174270ranking_model_174272ranking_model_174274ranking_model_174276ranking_model_174278ranking_model_174280ranking_model_174282ranking_model_174284ranking_model_174286ranking_model_174288*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_ranking_model_layer_call_and_return_conditional_losses_173974}
IdentityIdentity.ranking_model/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????n
NoOpNoOp&^ranking_model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????:?????????: : : : : : : : : : : : : : : : 2N
%ranking_model/StatefulPartitionedCall%ranking_model/StatefulPartitionedCall:M I
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:

_output_shapes
: :

_output_shapes
: 
?	
?
C__inference_dense_4_layer_call_and_return_conditional_losses_173685

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
-
__inference__destroyer_175360
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
&__inference_dense_layer_call_fn_175252

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_173618p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
-__inference_sequential_2_layer_call_fn_175135

inputs
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_173821o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_<lambda>_1753946
2key_value_init112_lookuptableimportv2_table_handle.
*key_value_init112_lookuptableimportv2_keys0
,key_value_init112_lookuptableimportv2_values	
identity??%key_value_init112/LookupTableImportV2?
%key_value_init112/LookupTableImportV2LookupTableImportV22key_value_init112_lookuptableimportv2_table_handle*key_value_init112_lookuptableimportv2_keys,key_value_init112_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: n
NoOpNoOp&^key_value_init112/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :7:72N
%key_value_init112/LookupTableImportV2%key_value_init112/LookupTableImportV2: 

_output_shapes
:7: 

_output_shapes
:7
?
?
I__inference_ranking_model_layer_call_and_return_conditional_losses_174207
input_1
input_2
sequential_174169
sequential_174171	$
sequential_174173:	?@
sequential_1_174176
sequential_1_174178	%
sequential_1_174180:8@'
sequential_2_174185:
??"
sequential_2_174187:	?&
sequential_2_174189:	?@!
sequential_2_174191:@%
sequential_2_174193:@ !
sequential_2_174195: %
sequential_2_174197: !
sequential_2_174199:%
sequential_2_174201:!
sequential_2_174203:
identity??"sequential/StatefulPartitionedCall?$sequential_1/StatefulPartitionedCall?$sequential_2/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_174169sequential_174171sequential_174173*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_173409?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallinput_2sequential_1_174176sequential_1_174178sequential_1_174180*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_173517M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2+sequential/StatefulPartitionedCall:output:0-sequential_1/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0sequential_2_174185sequential_2_174187sequential_2_174189sequential_2_174191sequential_2_174193sequential_2_174195sequential_2_174197sequential_2_174199sequential_2_174201sequential_2_174203*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_173692|
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????:?????????: : : : : : : : : : : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:LH
#
_output_shapes
:?????????
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: 
?
;
__inference__creator_175365
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name113*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
E__inference_embedding_layer_call_and_return_conditional_losses_175227

inputs	*
embedding_lookup_175221:	?@
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_175221inputs*
Tindices0	**
_class 
loc:@embedding_lookup/175221*'
_output_shapes
:?????????@*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/175221*'
_output_shapes
:?????????@}
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????@Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?h
?
P__inference_supplier_recommender_layer_call_and_return_conditional_losses_174795
features_subject_id
features_supplier_id[
Wranking_model_sequential_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle\
Xranking_model_sequential_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	M
:ranking_model_sequential_embedding_embedding_lookup_174744:	?@_
[ranking_model_sequential_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handle`
\ranking_model_sequential_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_default_value	P
>ranking_model_sequential_1_embedding_1_embedding_lookup_174753:8@S
?ranking_model_sequential_2_dense_matmul_readvariableop_resource:
??O
@ranking_model_sequential_2_dense_biasadd_readvariableop_resource:	?T
Aranking_model_sequential_2_dense_1_matmul_readvariableop_resource:	?@P
Branking_model_sequential_2_dense_1_biasadd_readvariableop_resource:@S
Aranking_model_sequential_2_dense_2_matmul_readvariableop_resource:@ P
Branking_model_sequential_2_dense_2_biasadd_readvariableop_resource: S
Aranking_model_sequential_2_dense_3_matmul_readvariableop_resource: P
Branking_model_sequential_2_dense_3_biasadd_readvariableop_resource:S
Aranking_model_sequential_2_dense_4_matmul_readvariableop_resource:P
Branking_model_sequential_2_dense_4_biasadd_readvariableop_resource:
identity??3ranking_model/sequential/embedding/embedding_lookup?Jranking_model/sequential/string_lookup/hash_table_Lookup/LookupTableFindV2?7ranking_model/sequential_1/embedding_1/embedding_lookup?Nranking_model/sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2?7ranking_model/sequential_2/dense/BiasAdd/ReadVariableOp?6ranking_model/sequential_2/dense/MatMul/ReadVariableOp?9ranking_model/sequential_2/dense_1/BiasAdd/ReadVariableOp?8ranking_model/sequential_2/dense_1/MatMul/ReadVariableOp?9ranking_model/sequential_2/dense_2/BiasAdd/ReadVariableOp?8ranking_model/sequential_2/dense_2/MatMul/ReadVariableOp?9ranking_model/sequential_2/dense_3/BiasAdd/ReadVariableOp?8ranking_model/sequential_2/dense_3/MatMul/ReadVariableOp?9ranking_model/sequential_2/dense_4/BiasAdd/ReadVariableOp?8ranking_model/sequential_2/dense_4/MatMul/ReadVariableOp?
Jranking_model/sequential/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Wranking_model_sequential_string_lookup_hash_table_lookup_lookuptablefindv2_table_handlefeatures_supplier_idXranking_model_sequential_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
/ranking_model/sequential/string_lookup/IdentityIdentitySranking_model/sequential/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
3ranking_model/sequential/embedding/embedding_lookupResourceGather:ranking_model_sequential_embedding_embedding_lookup_1747448ranking_model/sequential/string_lookup/Identity:output:0*
Tindices0	*M
_classC
A?loc:@ranking_model/sequential/embedding/embedding_lookup/174744*'
_output_shapes
:?????????@*
dtype0?
<ranking_model/sequential/embedding/embedding_lookup/IdentityIdentity<ranking_model/sequential/embedding/embedding_lookup:output:0*
T0*M
_classC
A?loc:@ranking_model/sequential/embedding/embedding_lookup/174744*'
_output_shapes
:?????????@?
>ranking_model/sequential/embedding/embedding_lookup/Identity_1IdentityEranking_model/sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@?
Nranking_model/sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2[ranking_model_sequential_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handlefeatures_subject_id\ranking_model_sequential_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
3ranking_model/sequential_1/string_lookup_1/IdentityIdentityWranking_model/sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
7ranking_model/sequential_1/embedding_1/embedding_lookupResourceGather>ranking_model_sequential_1_embedding_1_embedding_lookup_174753<ranking_model/sequential_1/string_lookup_1/Identity:output:0*
Tindices0	*Q
_classG
ECloc:@ranking_model/sequential_1/embedding_1/embedding_lookup/174753*'
_output_shapes
:?????????@*
dtype0?
@ranking_model/sequential_1/embedding_1/embedding_lookup/IdentityIdentity@ranking_model/sequential_1/embedding_1/embedding_lookup:output:0*
T0*Q
_classG
ECloc:@ranking_model/sequential_1/embedding_1/embedding_lookup/174753*'
_output_shapes
:?????????@?
Branking_model/sequential_1/embedding_1/embedding_lookup/Identity_1IdentityIranking_model/sequential_1/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@[
ranking_model/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
ranking_model/concatConcatV2Granking_model/sequential/embedding/embedding_lookup/Identity_1:output:0Kranking_model/sequential_1/embedding_1/embedding_lookup/Identity_1:output:0"ranking_model/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
6ranking_model/sequential_2/dense/MatMul/ReadVariableOpReadVariableOp?ranking_model_sequential_2_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
'ranking_model/sequential_2/dense/MatMulMatMulranking_model/concat:output:0>ranking_model/sequential_2/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
7ranking_model/sequential_2/dense/BiasAdd/ReadVariableOpReadVariableOp@ranking_model_sequential_2_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
(ranking_model/sequential_2/dense/BiasAddBiasAdd1ranking_model/sequential_2/dense/MatMul:product:0?ranking_model/sequential_2/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
%ranking_model/sequential_2/dense/ReluRelu1ranking_model/sequential_2/dense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
8ranking_model/sequential_2/dense_1/MatMul/ReadVariableOpReadVariableOpAranking_model_sequential_2_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
)ranking_model/sequential_2/dense_1/MatMulMatMul3ranking_model/sequential_2/dense/Relu:activations:0@ranking_model/sequential_2/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
9ranking_model/sequential_2/dense_1/BiasAdd/ReadVariableOpReadVariableOpBranking_model_sequential_2_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
*ranking_model/sequential_2/dense_1/BiasAddBiasAdd3ranking_model/sequential_2/dense_1/MatMul:product:0Aranking_model/sequential_2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
,ranking_model/sequential_2/dense_1/LeakyRelu	LeakyRelu3ranking_model/sequential_2/dense_1/BiasAdd:output:0*'
_output_shapes
:?????????@?
8ranking_model/sequential_2/dense_2/MatMul/ReadVariableOpReadVariableOpAranking_model_sequential_2_dense_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0?
)ranking_model/sequential_2/dense_2/MatMulMatMul:ranking_model/sequential_2/dense_1/LeakyRelu:activations:0@ranking_model/sequential_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
9ranking_model/sequential_2/dense_2/BiasAdd/ReadVariableOpReadVariableOpBranking_model_sequential_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
*ranking_model/sequential_2/dense_2/BiasAddBiasAdd3ranking_model/sequential_2/dense_2/MatMul:product:0Aranking_model/sequential_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
,ranking_model/sequential_2/dense_2/LeakyRelu	LeakyRelu3ranking_model/sequential_2/dense_2/BiasAdd:output:0*'
_output_shapes
:????????? ?
8ranking_model/sequential_2/dense_3/MatMul/ReadVariableOpReadVariableOpAranking_model_sequential_2_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
)ranking_model/sequential_2/dense_3/MatMulMatMul:ranking_model/sequential_2/dense_2/LeakyRelu:activations:0@ranking_model/sequential_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
9ranking_model/sequential_2/dense_3/BiasAdd/ReadVariableOpReadVariableOpBranking_model_sequential_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
*ranking_model/sequential_2/dense_3/BiasAddBiasAdd3ranking_model/sequential_2/dense_3/MatMul:product:0Aranking_model/sequential_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
'ranking_model/sequential_2/dense_3/ReluRelu3ranking_model/sequential_2/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
8ranking_model/sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOpAranking_model_sequential_2_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
)ranking_model/sequential_2/dense_4/MatMulMatMul5ranking_model/sequential_2/dense_3/Relu:activations:0@ranking_model/sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
9ranking_model/sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOpBranking_model_sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
*ranking_model/sequential_2/dense_4/BiasAddBiasAdd3ranking_model/sequential_2/dense_4/MatMul:product:0Aranking_model/sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
IdentityIdentity3ranking_model/sequential_2/dense_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp4^ranking_model/sequential/embedding/embedding_lookupK^ranking_model/sequential/string_lookup/hash_table_Lookup/LookupTableFindV28^ranking_model/sequential_1/embedding_1/embedding_lookupO^ranking_model/sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV28^ranking_model/sequential_2/dense/BiasAdd/ReadVariableOp7^ranking_model/sequential_2/dense/MatMul/ReadVariableOp:^ranking_model/sequential_2/dense_1/BiasAdd/ReadVariableOp9^ranking_model/sequential_2/dense_1/MatMul/ReadVariableOp:^ranking_model/sequential_2/dense_2/BiasAdd/ReadVariableOp9^ranking_model/sequential_2/dense_2/MatMul/ReadVariableOp:^ranking_model/sequential_2/dense_3/BiasAdd/ReadVariableOp9^ranking_model/sequential_2/dense_3/MatMul/ReadVariableOp:^ranking_model/sequential_2/dense_4/BiasAdd/ReadVariableOp9^ranking_model/sequential_2/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????:?????????: : : : : : : : : : : : : : : : 2j
3ranking_model/sequential/embedding/embedding_lookup3ranking_model/sequential/embedding/embedding_lookup2?
Jranking_model/sequential/string_lookup/hash_table_Lookup/LookupTableFindV2Jranking_model/sequential/string_lookup/hash_table_Lookup/LookupTableFindV22r
7ranking_model/sequential_1/embedding_1/embedding_lookup7ranking_model/sequential_1/embedding_1/embedding_lookup2?
Nranking_model/sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2Nranking_model/sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV22r
7ranking_model/sequential_2/dense/BiasAdd/ReadVariableOp7ranking_model/sequential_2/dense/BiasAdd/ReadVariableOp2p
6ranking_model/sequential_2/dense/MatMul/ReadVariableOp6ranking_model/sequential_2/dense/MatMul/ReadVariableOp2v
9ranking_model/sequential_2/dense_1/BiasAdd/ReadVariableOp9ranking_model/sequential_2/dense_1/BiasAdd/ReadVariableOp2t
8ranking_model/sequential_2/dense_1/MatMul/ReadVariableOp8ranking_model/sequential_2/dense_1/MatMul/ReadVariableOp2v
9ranking_model/sequential_2/dense_2/BiasAdd/ReadVariableOp9ranking_model/sequential_2/dense_2/BiasAdd/ReadVariableOp2t
8ranking_model/sequential_2/dense_2/MatMul/ReadVariableOp8ranking_model/sequential_2/dense_2/MatMul/ReadVariableOp2v
9ranking_model/sequential_2/dense_3/BiasAdd/ReadVariableOp9ranking_model/sequential_2/dense_3/BiasAdd/ReadVariableOp2t
8ranking_model/sequential_2/dense_3/MatMul/ReadVariableOp8ranking_model/sequential_2/dense_3/MatMul/ReadVariableOp2v
9ranking_model/sequential_2/dense_4/BiasAdd/ReadVariableOp9ranking_model/sequential_2/dense_4/BiasAdd/ReadVariableOp2t
8ranking_model/sequential_2/dense_4/MatMul/ReadVariableOp8ranking_model/sequential_2/dense_4/MatMul/ReadVariableOp:X T
#
_output_shapes
:?????????
-
_user_specified_namefeatures/subject_id:YU
#
_output_shapes
:?????????
.
_user_specified_namefeatures/supplier_id:

_output_shapes
: :

_output_shapes
: 
?
?
__inference_<lambda>_1753865
1key_value_init90_lookuptableimportv2_table_handle-
)key_value_init90_lookuptableimportv2_keys/
+key_value_init90_lookuptableimportv2_values	
identity??$key_value_init90/LookupTableImportV2?
$key_value_init90/LookupTableImportV2LookupTableImportV21key_value_init90_lookuptableimportv2_table_handle)key_value_init90_lookuptableimportv2_keys+key_value_init90_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: m
NoOpNoOp%^key_value_init90/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2L
$key_value_init90/LookupTableImportV2$key_value_init90/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?V
?
I__inference_ranking_model_layer_call_and_return_conditional_losses_174989
inputs_0
inputs_1M
Isequential_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleN
Jsequential_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	?
,sequential_embedding_embedding_lookup_174938:	?@Q
Msequential_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleR
Nsequential_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_default_value	B
0sequential_1_embedding_1_embedding_lookup_174947:8@E
1sequential_2_dense_matmul_readvariableop_resource:
??A
2sequential_2_dense_biasadd_readvariableop_resource:	?F
3sequential_2_dense_1_matmul_readvariableop_resource:	?@B
4sequential_2_dense_1_biasadd_readvariableop_resource:@E
3sequential_2_dense_2_matmul_readvariableop_resource:@ B
4sequential_2_dense_2_biasadd_readvariableop_resource: E
3sequential_2_dense_3_matmul_readvariableop_resource: B
4sequential_2_dense_3_biasadd_readvariableop_resource:E
3sequential_2_dense_4_matmul_readvariableop_resource:B
4sequential_2_dense_4_biasadd_readvariableop_resource:
identity??%sequential/embedding/embedding_lookup?<sequential/string_lookup/hash_table_Lookup/LookupTableFindV2?)sequential_1/embedding_1/embedding_lookup?@sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2?)sequential_2/dense/BiasAdd/ReadVariableOp?(sequential_2/dense/MatMul/ReadVariableOp?+sequential_2/dense_1/BiasAdd/ReadVariableOp?*sequential_2/dense_1/MatMul/ReadVariableOp?+sequential_2/dense_2/BiasAdd/ReadVariableOp?*sequential_2/dense_2/MatMul/ReadVariableOp?+sequential_2/dense_3/BiasAdd/ReadVariableOp?*sequential_2/dense_3/MatMul/ReadVariableOp?+sequential_2/dense_4/BiasAdd/ReadVariableOp?*sequential_2/dense_4/MatMul/ReadVariableOp?
<sequential/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Isequential_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleinputs_0Jsequential_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
!sequential/string_lookup/IdentityIdentityEsequential/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
%sequential/embedding/embedding_lookupResourceGather,sequential_embedding_embedding_lookup_174938*sequential/string_lookup/Identity:output:0*
Tindices0	*?
_class5
31loc:@sequential/embedding/embedding_lookup/174938*'
_output_shapes
:?????????@*
dtype0?
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0*
T0*?
_class5
31loc:@sequential/embedding/embedding_lookup/174938*'
_output_shapes
:?????????@?
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@?
@sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Msequential_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleinputs_1Nsequential_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
%sequential_1/string_lookup_1/IdentityIdentityIsequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)sequential_1/embedding_1/embedding_lookupResourceGather0sequential_1_embedding_1_embedding_lookup_174947.sequential_1/string_lookup_1/Identity:output:0*
Tindices0	*C
_class9
75loc:@sequential_1/embedding_1/embedding_lookup/174947*'
_output_shapes
:?????????@*
dtype0?
2sequential_1/embedding_1/embedding_lookup/IdentityIdentity2sequential_1/embedding_1/embedding_lookup:output:0*
T0*C
_class9
75loc:@sequential_1/embedding_1/embedding_lookup/174947*'
_output_shapes
:?????????@?
4sequential_1/embedding_1/embedding_lookup/Identity_1Identity;sequential_1/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV29sequential/embedding/embedding_lookup/Identity_1:output:0=sequential_1/embedding_1/embedding_lookup/Identity_1:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
(sequential_2/dense/MatMul/ReadVariableOpReadVariableOp1sequential_2_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential_2/dense/MatMulMatMulconcat:output:00sequential_2/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
)sequential_2/dense/BiasAdd/ReadVariableOpReadVariableOp2sequential_2_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_2/dense/BiasAddBiasAdd#sequential_2/dense/MatMul:product:01sequential_2/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????w
sequential_2/dense/ReluRelu#sequential_2/dense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
*sequential_2/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
sequential_2/dense_1/MatMulMatMul%sequential_2/dense/Relu:activations:02sequential_2/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
+sequential_2/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_2/dense_1/BiasAddBiasAdd%sequential_2/dense_1/MatMul:product:03sequential_2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@{
sequential_2/dense_1/LeakyRelu	LeakyRelu%sequential_2/dense_1/BiasAdd:output:0*'
_output_shapes
:?????????@?
*sequential_2/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0?
sequential_2/dense_2/MatMulMatMul,sequential_2/dense_1/LeakyRelu:activations:02sequential_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
+sequential_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_2/dense_2/BiasAddBiasAdd%sequential_2/dense_2/MatMul:product:03sequential_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? {
sequential_2/dense_2/LeakyRelu	LeakyRelu%sequential_2/dense_2/BiasAdd:output:0*'
_output_shapes
:????????? ?
*sequential_2/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
sequential_2/dense_3/MatMulMatMul,sequential_2/dense_2/LeakyRelu:activations:02sequential_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_2/dense_3/BiasAddBiasAdd%sequential_2/dense_3/MatMul:product:03sequential_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
sequential_2/dense_3/ReluRelu%sequential_2/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
*sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
sequential_2/dense_4/MatMulMatMul'sequential_2/dense_3/Relu:activations:02sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_2/dense_4/BiasAddBiasAdd%sequential_2/dense_4/MatMul:product:03sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
IdentityIdentity%sequential_2/dense_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp&^sequential/embedding/embedding_lookup=^sequential/string_lookup/hash_table_Lookup/LookupTableFindV2*^sequential_1/embedding_1/embedding_lookupA^sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2*^sequential_2/dense/BiasAdd/ReadVariableOp)^sequential_2/dense/MatMul/ReadVariableOp,^sequential_2/dense_1/BiasAdd/ReadVariableOp+^sequential_2/dense_1/MatMul/ReadVariableOp,^sequential_2/dense_2/BiasAdd/ReadVariableOp+^sequential_2/dense_2/MatMul/ReadVariableOp,^sequential_2/dense_3/BiasAdd/ReadVariableOp+^sequential_2/dense_3/MatMul/ReadVariableOp,^sequential_2/dense_4/BiasAdd/ReadVariableOp+^sequential_2/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????:?????????: : : : : : : : : : : : : : : : 2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2|
<sequential/string_lookup/hash_table_Lookup/LookupTableFindV2<sequential/string_lookup/hash_table_Lookup/LookupTableFindV22V
)sequential_1/embedding_1/embedding_lookup)sequential_1/embedding_1/embedding_lookup2?
@sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2@sequential_1/string_lookup_1/hash_table_Lookup/LookupTableFindV22V
)sequential_2/dense/BiasAdd/ReadVariableOp)sequential_2/dense/BiasAdd/ReadVariableOp2T
(sequential_2/dense/MatMul/ReadVariableOp(sequential_2/dense/MatMul/ReadVariableOp2Z
+sequential_2/dense_1/BiasAdd/ReadVariableOp+sequential_2/dense_1/BiasAdd/ReadVariableOp2X
*sequential_2/dense_1/MatMul/ReadVariableOp*sequential_2/dense_1/MatMul/ReadVariableOp2Z
+sequential_2/dense_2/BiasAdd/ReadVariableOp+sequential_2/dense_2/BiasAdd/ReadVariableOp2X
*sequential_2/dense_2/MatMul/ReadVariableOp*sequential_2/dense_2/MatMul/ReadVariableOp2Z
+sequential_2/dense_3/BiasAdd/ReadVariableOp+sequential_2/dense_3/BiasAdd/ReadVariableOp2X
*sequential_2/dense_3/MatMul/ReadVariableOp*sequential_2/dense_3/MatMul/ReadVariableOp2Z
+sequential_2/dense_4/BiasAdd/ReadVariableOp+sequential_2/dense_4/BiasAdd/ReadVariableOp2X
*sequential_2/dense_4/MatMul/ReadVariableOp*sequential_2/dense_4/MatMul/ReadVariableOp:M I
#
_output_shapes
:?????????
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: 
?
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_173600
string_lookup_1_inputD
@string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleE
Astring_lookup_1_hash_table_lookup_lookuptablefindv2_default_value	$
embedding_1_173596:8@
identity??#embedding_1/StatefulPartitionedCall?3string_lookup_1/hash_table_Lookup/LookupTableFindV2?
3string_lookup_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2@string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handlestring_lookup_1_inputAstring_lookup_1_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup_1/IdentityIdentity<string_lookup_1/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_1/Identity:output:0embedding_1_173596*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_173512{
IdentityIdentity,embedding_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp$^embedding_1/StatefulPartitionedCall4^string_lookup_1/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2j
3string_lookup_1/hash_table_Lookup/LookupTableFindV23string_lookup_1/hash_table_Lookup/LookupTableFindV2:Z V
#
_output_shapes
:?????????
/
_user_specified_namestring_lookup_1_input:

_output_shapes
: 
?
?
,__inference_embedding_1_layer_call_fn_175234

inputs	
unknown:8@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_173512o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_173589
string_lookup_1_inputD
@string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleE
Astring_lookup_1_hash_table_lookup_lookuptablefindv2_default_value	$
embedding_1_173585:8@
identity??#embedding_1/StatefulPartitionedCall?3string_lookup_1/hash_table_Lookup/LookupTableFindV2?
3string_lookup_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2@string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handlestring_lookup_1_inputAstring_lookup_1_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup_1/IdentityIdentity<string_lookup_1/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_1/Identity:output:0embedding_1_173585*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_173512{
IdentityIdentity,embedding_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp$^embedding_1/StatefulPartitionedCall4^string_lookup_1/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2j
3string_lookup_1/hash_table_Lookup/LookupTableFindV23string_lookup_1/hash_table_Lookup/LookupTableFindV2:Z V
#
_output_shapes
:?????????
/
_user_specified_namestring_lookup_1_input:

_output_shapes
: 
?
?
5__inference_supplier_recommender_layer_call_fn_174677
features_subject_id
features_supplier_id
unknown
	unknown_0	
	unknown_1:	?@
	unknown_2
	unknown_3	
	unknown_4:8@
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?@
	unknown_8:@
	unknown_9:@ 

unknown_10: 

unknown_11: 

unknown_12:

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallfeatures_subject_idfeatures_supplier_idunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_supplier_recommender_layer_call_and_return_conditional_losses_174406o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
#
_output_shapes
:?????????
-
_user_specified_namefeatures/subject_id:YU
#
_output_shapes
:?????????
.
_user_specified_namefeatures/supplier_id:

_output_shapes
: :

_output_shapes
: 
?
;
__inference__creator_175347
identity??
hash_tablej

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name91*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_175024

inputsB
>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleC
?string_lookup_hash_table_lookup_lookuptablefindv2_default_value	4
!embedding_embedding_lookup_175018:	?@
identity??embedding/embedding_lookup?1string_lookup/hash_table_Lookup/LookupTableFindV2?
1string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleinputs?string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup/IdentityIdentity:string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_175018string_lookup/Identity:output:0*
Tindices0	*4
_class*
(&loc:@embedding/embedding_lookup/175018*'
_output_shapes
:?????????@*
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/175018*'
_output_shapes
:?????????@?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@}
IdentityIdentity.embedding/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^embedding/embedding_lookup2^string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 28
embedding/embedding_lookupembedding/embedding_lookup2f
1string_lookup/hash_table_Lookup/LookupTableFindV21string_lookup/hash_table_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_173517

inputsD
@string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleE
Astring_lookup_1_hash_table_lookup_lookuptablefindv2_default_value	$
embedding_1_173513:8@
identity??#embedding_1/StatefulPartitionedCall?3string_lookup_1/hash_table_Lookup/LookupTableFindV2?
3string_lookup_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2@string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleinputsAstring_lookup_1_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup_1/IdentityIdentity<string_lookup_1/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_1/Identity:output:0embedding_1_173513*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_173512{
IdentityIdentity,embedding_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp$^embedding_1/StatefulPartitionedCall4^string_lookup_1/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2j
3string_lookup_1/hash_table_Lookup/LookupTableFindV23string_lookup_1/hash_table_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
??
?
"__inference__traced_restore_175696
file_prefix8
%assignvariableop_embedding_embeddings:	?@;
)assignvariableop_1_embedding_1_embeddings:8@3
assignvariableop_2_dense_kernel:
??,
assignvariableop_3_dense_bias:	?4
!assignvariableop_4_dense_1_kernel:	?@-
assignvariableop_5_dense_1_bias:@3
!assignvariableop_6_dense_2_kernel:@ -
assignvariableop_7_dense_2_bias: 3
!assignvariableop_8_dense_3_kernel: -
assignvariableop_9_dense_3_bias:4
"assignvariableop_10_dense_4_kernel:.
 assignvariableop_11_dense_4_bias:'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: #
assignvariableop_17_total: #
assignvariableop_18_count: B
/assignvariableop_19_adam_embedding_embeddings_m:	?@C
1assignvariableop_20_adam_embedding_1_embeddings_m:8@;
'assignvariableop_21_adam_dense_kernel_m:
??4
%assignvariableop_22_adam_dense_bias_m:	?<
)assignvariableop_23_adam_dense_1_kernel_m:	?@5
'assignvariableop_24_adam_dense_1_bias_m:@;
)assignvariableop_25_adam_dense_2_kernel_m:@ 5
'assignvariableop_26_adam_dense_2_bias_m: ;
)assignvariableop_27_adam_dense_3_kernel_m: 5
'assignvariableop_28_adam_dense_3_bias_m:;
)assignvariableop_29_adam_dense_4_kernel_m:5
'assignvariableop_30_adam_dense_4_bias_m:B
/assignvariableop_31_adam_embedding_embeddings_v:	?@C
1assignvariableop_32_adam_embedding_1_embeddings_v:8@;
'assignvariableop_33_adam_dense_kernel_v:
??4
%assignvariableop_34_adam_dense_bias_v:	?<
)assignvariableop_35_adam_dense_1_kernel_v:	?@5
'assignvariableop_36_adam_dense_1_bias_v:@;
)assignvariableop_37_adam_dense_2_kernel_v:@ 5
'assignvariableop_38_adam_dense_2_bias_v: ;
)assignvariableop_39_adam_dense_3_kernel_v: 5
'assignvariableop_40_adam_dense_3_bias_v:;
)assignvariableop_41_adam_dense_4_kernel_v:5
'assignvariableop_42_adam_dense_4_bias_v:
identity_44??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*?
value?B?,B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp)assignvariableop_1_embedding_1_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_4_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_4_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp/assignvariableop_19_adam_embedding_embeddings_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp1assignvariableop_20_adam_embedding_1_embeddings_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp%assignvariableop_22_adam_dense_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_2_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_2_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_3_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_3_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_4_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_4_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp/assignvariableop_31_adam_embedding_embeddings_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp1assignvariableop_32_adam_embedding_1_embeddings_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp'assignvariableop_33_adam_dense_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp%assignvariableop_34_adam_dense_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_1_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_1_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_2_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_2_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_3_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_dense_3_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_4_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_4_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_44IdentityIdentity_43:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_44Identity_44:output:0*k
_input_shapesZ
X: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
-__inference_sequential_1_layer_call_fn_175059

inputs
unknown
	unknown_0	
	unknown_1:8@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_173558o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
H__inference_sequential_2_layer_call_and_return_conditional_losses_173898
dense_input 
dense_173872:
??
dense_173874:	?!
dense_1_173877:	?@
dense_1_173879:@ 
dense_2_173882:@ 
dense_2_173884:  
dense_3_173887: 
dense_3_173889: 
dense_4_173892:
dense_4_173894:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_173872dense_173874*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_173618?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_173877dense_1_173879*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_173635?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_173882dense_2_173884*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_173652?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_173887dense_3_173889*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_173669?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_173892dense_4_173894*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_173685w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:U Q
(
_output_shapes
:??????????
%
_user_specified_namedense_input
?
?
.__inference_ranking_model_layer_call_fn_174833
inputs_0
inputs_1
unknown
	unknown_0	
	unknown_1:	?@
	unknown_2
	unknown_3	
	unknown_4:8@
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?@
	unknown_8:@
	unknown_9:@ 

unknown_10: 

unknown_11: 

unknown_12:

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_ranking_model_layer_call_and_return_conditional_losses_173974o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:?????????
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: 
?

?
-__inference_sequential_2_layer_call_fn_175110

inputs
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_173692o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
A__inference_dense_layer_call_and_return_conditional_losses_175263

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference__initializer_1753736
2key_value_init112_lookuptableimportv2_table_handle.
*key_value_init112_lookuptableimportv2_keys0
,key_value_init112_lookuptableimportv2_values	
identity??%key_value_init112/LookupTableImportV2?
%key_value_init112/LookupTableImportV2LookupTableImportV22key_value_init112_lookuptableimportv2_table_handle*key_value_init112_lookuptableimportv2_keys,key_value_init112_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: n
NoOpNoOp&^key_value_init112/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :7:72N
%key_value_init112/LookupTableImportV2%key_value_init112/LookupTableImportV2: 

_output_shapes
:7: 

_output_shapes
:7
?
?
-__inference_sequential_1_layer_call_fn_173526
string_lookup_1_input
unknown
	unknown_0	
	unknown_1:8@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_1_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_173517o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
#
_output_shapes
:?????????
/
_user_specified_namestring_lookup_1_input:

_output_shapes
: 
?
?
.__inference_ranking_model_layer_call_fn_174165
input_1
input_2
unknown
	unknown_0	
	unknown_1:	?@
	unknown_2
	unknown_3	
	unknown_4:8@
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?@
	unknown_8:@
	unknown_9:@ 

unknown_10: 

unknown_11: 

unknown_12:

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_ranking_model_layer_call_and_return_conditional_losses_174092o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:LH
#
_output_shapes
:?????????
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: 
?
?
H__inference_sequential_2_layer_call_and_return_conditional_losses_173692

inputs 
dense_173619:
??
dense_173621:	?!
dense_1_173636:	?@
dense_1_173638:@ 
dense_2_173653:@ 
dense_2_173655:  
dense_3_173670: 
dense_3_173672: 
dense_4_173686:
dense_4_173688:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_173619dense_173621*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_173618?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_173636dense_1_173638*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_173635?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_173653dense_2_173655*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_173652?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_173670dense_3_173672*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_173669?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_173686dense_4_173688*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_173685w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
-
__inference__destroyer_175378
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
(__inference_dense_4_layer_call_fn_175332

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_173685o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_embedding_layer_call_and_return_conditional_losses_173404

inputs	*
embedding_lookup_173398:	?@
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_173398inputs*
Tindices0	**
_class 
loc:@embedding_lookup/173398*'
_output_shapes
:?????????@*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/173398*'
_output_shapes
:?????????@}
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????@Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_1_layer_call_fn_173578
string_lookup_1_input
unknown
	unknown_0	
	unknown_1:8@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_1_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_173558o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
#
_output_shapes
:?????????
/
_user_specified_namestring_lookup_1_input:

_output_shapes
: 
?

?
C__inference_dense_3_layer_call_and_return_conditional_losses_173669

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
C__inference_dense_4_layer_call_and_return_conditional_losses_175342

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_dense_2_layer_call_fn_175292

inputs
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_173652o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

*__inference_embedding_layer_call_fn_175218

inputs	
unknown:	?@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_173404o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
P__inference_supplier_recommender_layer_call_and_return_conditional_losses_174517

subject_id
supplier_id
ranking_model_174483
ranking_model_174485	'
ranking_model_174487:	?@
ranking_model_174489
ranking_model_174491	&
ranking_model_174493:8@(
ranking_model_174495:
??#
ranking_model_174497:	?'
ranking_model_174499:	?@"
ranking_model_174501:@&
ranking_model_174503:@ "
ranking_model_174505: &
ranking_model_174507: "
ranking_model_174509:&
ranking_model_174511:"
ranking_model_174513:
identity??%ranking_model/StatefulPartitionedCall?
%ranking_model/StatefulPartitionedCallStatefulPartitionedCallsupplier_id
subject_idranking_model_174483ranking_model_174485ranking_model_174487ranking_model_174489ranking_model_174491ranking_model_174493ranking_model_174495ranking_model_174497ranking_model_174499ranking_model_174501ranking_model_174503ranking_model_174505ranking_model_174507ranking_model_174509ranking_model_174511ranking_model_174513*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_ranking_model_layer_call_and_return_conditional_losses_173974}
IdentityIdentity.ranking_model/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????n
NoOpNoOp&^ranking_model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????:?????????: : : : : : : : : : : : : : : : 2N
%ranking_model/StatefulPartitionedCall%ranking_model/StatefulPartitionedCall:O K
#
_output_shapes
:?????????
$
_user_specified_name
subject_id:PL
#
_output_shapes
:?????????
%
_user_specified_namesupplier_id:

_output_shapes
: :

_output_shapes
: 
?
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_175085

inputsD
@string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleE
Astring_lookup_1_hash_table_lookup_lookuptablefindv2_default_value	5
#embedding_1_embedding_lookup_175079:8@
identity??embedding_1/embedding_lookup?3string_lookup_1/hash_table_Lookup/LookupTableFindV2?
3string_lookup_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2@string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleinputsAstring_lookup_1_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup_1/IdentityIdentity<string_lookup_1/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
embedding_1/embedding_lookupResourceGather#embedding_1_embedding_lookup_175079!string_lookup_1/Identity:output:0*
Tindices0	*6
_class,
*(loc:@embedding_1/embedding_lookup/175079*'
_output_shapes
:?????????@*
dtype0?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_1/embedding_lookup/175079*'
_output_shapes
:?????????@?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@
IdentityIdentity0embedding_1/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^embedding_1/embedding_lookup4^string_lookup_1/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2j
3string_lookup_1/hash_table_Lookup/LookupTableFindV23string_lookup_1/hash_table_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
H__inference_sequential_2_layer_call_and_return_conditional_losses_173927
dense_input 
dense_173901:
??
dense_173903:	?!
dense_1_173906:	?@
dense_1_173908:@ 
dense_2_173911:@ 
dense_2_173913:  
dense_3_173916: 
dense_3_173918: 
dense_4_173921:
dense_4_173923:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_173901dense_173903*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_173618?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_173906dense_1_173908*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_173635?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_173911dense_2_173913*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_173652?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_173916dense_3_173918*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_173669?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_173921dense_4_173923*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_173685w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:U Q
(
_output_shapes
:??????????
%
_user_specified_namedense_input
?
?
5__inference_supplier_recommender_layer_call_fn_174327

subject_id
supplier_id
unknown
	unknown_0	
	unknown_1:	?@
	unknown_2
	unknown_3	
	unknown_4:8@
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?@
	unknown_8:@
	unknown_9:@ 

unknown_10: 

unknown_11: 

unknown_12:

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
subject_idsupplier_idunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_supplier_recommender_layer_call_and_return_conditional_losses_174292o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:?????????
$
_user_specified_name
subject_id:PL
#
_output_shapes
:?????????
%
_user_specified_namesupplier_id:

_output_shapes
: :

_output_shapes
: 
?
?
I__inference_ranking_model_layer_call_and_return_conditional_losses_174249
input_1
input_2
sequential_174211
sequential_174213	$
sequential_174215:	?@
sequential_1_174218
sequential_1_174220	%
sequential_1_174222:8@'
sequential_2_174227:
??"
sequential_2_174229:	?&
sequential_2_174231:	?@!
sequential_2_174233:@%
sequential_2_174235:@ !
sequential_2_174237: %
sequential_2_174239: !
sequential_2_174241:%
sequential_2_174243:!
sequential_2_174245:
identity??"sequential/StatefulPartitionedCall?$sequential_1/StatefulPartitionedCall?$sequential_2/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_174211sequential_174213sequential_174215*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_173450?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallinput_2sequential_1_174218sequential_1_174220sequential_1_174222*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_173558M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2+sequential/StatefulPartitionedCall:output:0-sequential_1/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0sequential_2_174227sequential_2_174229sequential_2_174231sequential_2_174233sequential_2_174235sequential_2_174237sequential_2_174239sequential_2_174241sequential_2_174243sequential_2_174245*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_173821|
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????:?????????: : : : : : : : : : : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:LH
#
_output_shapes
:?????????
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_sequential_layer_call_fn_175000

inputs
unknown
	unknown_0	
	unknown_1:	?@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_173409o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?

?
C__inference_dense_1_layer_call_and_return_conditional_losses_175283

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@Q
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:?????????@f
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_175072

inputsD
@string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleE
Astring_lookup_1_hash_table_lookup_lookuptablefindv2_default_value	5
#embedding_1_embedding_lookup_175066:8@
identity??embedding_1/embedding_lookup?3string_lookup_1/hash_table_Lookup/LookupTableFindV2?
3string_lookup_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2@string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleinputsAstring_lookup_1_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup_1/IdentityIdentity<string_lookup_1/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
embedding_1/embedding_lookupResourceGather#embedding_1_embedding_lookup_175066!string_lookup_1/Identity:output:0*
Tindices0	*6
_class,
*(loc:@embedding_1/embedding_lookup/175066*'
_output_shapes
:?????????@*
dtype0?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_1/embedding_lookup/175066*'
_output_shapes
:?????????@?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@
IdentityIdentity0embedding_1/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^embedding_1/embedding_lookup4^string_lookup_1/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2j
3string_lookup_1/hash_table_Lookup/LookupTableFindV23string_lookup_1/hash_table_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_173481
string_lookup_inputB
>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleC
?string_lookup_hash_table_lookup_lookuptablefindv2_default_value	#
embedding_173477:	?@
identity??!embedding/StatefulPartitionedCall?1string_lookup/hash_table_Lookup/LookupTableFindV2?
1string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2>string_lookup_hash_table_lookup_lookuptablefindv2_table_handlestring_lookup_input?string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup/IdentityIdentity:string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
!embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_173477*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_173404y
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp"^embedding/StatefulPartitionedCall2^string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2f
1string_lookup/hash_table_Lookup/LookupTableFindV21string_lookup/hash_table_Lookup/LookupTableFindV2:X T
#
_output_shapes
:?????????
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
?

?
C__inference_dense_1_layer_call_and_return_conditional_losses_173635

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@Q
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:?????????@f
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
P__inference_supplier_recommender_layer_call_and_return_conditional_losses_174555

subject_id
supplier_id
ranking_model_174521
ranking_model_174523	'
ranking_model_174525:	?@
ranking_model_174527
ranking_model_174529	&
ranking_model_174531:8@(
ranking_model_174533:
??#
ranking_model_174535:	?'
ranking_model_174537:	?@"
ranking_model_174539:@&
ranking_model_174541:@ "
ranking_model_174543: &
ranking_model_174545: "
ranking_model_174547:&
ranking_model_174549:"
ranking_model_174551:
identity??%ranking_model/StatefulPartitionedCall?
%ranking_model/StatefulPartitionedCallStatefulPartitionedCallsupplier_id
subject_idranking_model_174521ranking_model_174523ranking_model_174525ranking_model_174527ranking_model_174529ranking_model_174531ranking_model_174533ranking_model_174535ranking_model_174537ranking_model_174539ranking_model_174541ranking_model_174543ranking_model_174545ranking_model_174547ranking_model_174549ranking_model_174551*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_ranking_model_layer_call_and_return_conditional_losses_174092}
IdentityIdentity.ranking_model/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????n
NoOpNoOp&^ranking_model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????:?????????: : : : : : : : : : : : : : : : 2N
%ranking_model/StatefulPartitionedCall%ranking_model/StatefulPartitionedCall:O K
#
_output_shapes
:?????????
$
_user_specified_name
subject_id:PL
#
_output_shapes
:?????????
%
_user_specified_namesupplier_id:

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_embedding_1_layer_call_and_return_conditional_losses_173512

inputs	)
embedding_lookup_173506:8@
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_173506inputs*
Tindices0	**
_class 
loc:@embedding_lookup/173506*'
_output_shapes
:?????????@*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/173506*'
_output_shapes
:?????????@}
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????@Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_1_layer_call_fn_175048

inputs
unknown
	unknown_0	
	unknown_1:8@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_173517o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
I__inference_ranking_model_layer_call_and_return_conditional_losses_174092

inputs
inputs_1
sequential_174054
sequential_174056	$
sequential_174058:	?@
sequential_1_174061
sequential_1_174063	%
sequential_1_174065:8@'
sequential_2_174070:
??"
sequential_2_174072:	?&
sequential_2_174074:	?@!
sequential_2_174076:@%
sequential_2_174078:@ !
sequential_2_174080: %
sequential_2_174082: !
sequential_2_174084:%
sequential_2_174086:!
sequential_2_174088:
identity??"sequential/StatefulPartitionedCall?$sequential_1/StatefulPartitionedCall?$sequential_2/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_174054sequential_174056sequential_174058*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_173450?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1sequential_1_174061sequential_1_174063sequential_1_174065*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_173558M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2+sequential/StatefulPartitionedCall:output:0-sequential_1/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0sequential_2_174070sequential_2_174072sequential_2_174074sequential_2_174076sequential_2_174078sequential_2_174080sequential_2_174082sequential_2_174084sequential_2_174086sequential_2_174088*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_173821|
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????:?????????: : : : : : : : : : : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?

?
-__inference_sequential_2_layer_call_fn_173715
dense_input
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_173692o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:??????????
%
_user_specified_namedense_input"?L
saver_filename:0StatefulPartitionedCall_3:0StatefulPartitionedCall_48"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
=

subject_id/
serving_default_subject_id:0?????????
?
supplier_id0
serving_default_supplier_id:0?????????>
output_12
StatefulPartitionedCall_2:0?????????tensorflow/serving/predict:Ǔ
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
ranking_model
	task

	optimizer
loss

signatures"
_tf_keras_model
v
0
1
2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
trace_0
trace_1
 trace_2
!trace_32?
5__inference_supplier_recommender_layer_call_fn_174327
5__inference_supplier_recommender_layer_call_fn_174639
5__inference_supplier_recommender_layer_call_fn_174677
5__inference_supplier_recommender_layer_call_fn_174479?
???
FullArgSpec+
args#? 
jself

jfeatures

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ztrace_0ztrace_1z trace_2z!trace_3
?
"trace_0
#trace_1
$trace_2
%trace_32?
P__inference_supplier_recommender_layer_call_and_return_conditional_losses_174736
P__inference_supplier_recommender_layer_call_and_return_conditional_losses_174795
P__inference_supplier_recommender_layer_call_and_return_conditional_losses_174517
P__inference_supplier_recommender_layer_call_and_return_conditional_losses_174555?
???
FullArgSpec+
args#? 
jself

jfeatures

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z"trace_0z#trace_1z$trace_2z%trace_3
?B?
!__inference__wrapped_model_173384
subject_idsupplier_id"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,supplier_embeddings
-subject_embeddings
.ratings"
_tf_keras_model
?
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
5_ranking_metrics
6_prediction_metrics
7_label_metrics
8_loss_metrics"
_tf_keras_layer
?
9iter

:beta_1

;beta_2
	<decay
=learning_ratem?m?m?m?m?m?m?m?m?m?m?m?v?v?v?v?v?v?v?v?v?v?v?v?"
	optimizer
 "
trackable_dict_wrapper
,
>serving_default"
signature_map
':%	?@2embedding/embeddings
(:&8@2embedding_1/embeddings
 :
??2dense/kernel
:?2
dense/bias
!:	?@2dense_1/kernel
:@2dense_1/bias
 :@ 2dense_2/kernel
: 2dense_2/bias
 : 2dense_3/kernel
:2dense_3/bias
 :2dense_4/kernel
:2dense_4/bias
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
'
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
5__inference_supplier_recommender_layer_call_fn_174327
subject_idsupplier_id"?
???
FullArgSpec+
args#? 
jself

jfeatures

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
5__inference_supplier_recommender_layer_call_fn_174639features/subject_idfeatures/supplier_id"?
???
FullArgSpec+
args#? 
jself

jfeatures

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
5__inference_supplier_recommender_layer_call_fn_174677features/subject_idfeatures/supplier_id"?
???
FullArgSpec+
args#? 
jself

jfeatures

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
5__inference_supplier_recommender_layer_call_fn_174479
subject_idsupplier_id"?
???
FullArgSpec+
args#? 
jself

jfeatures

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
P__inference_supplier_recommender_layer_call_and_return_conditional_losses_174736features/subject_idfeatures/supplier_id"?
???
FullArgSpec+
args#? 
jself

jfeatures

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
P__inference_supplier_recommender_layer_call_and_return_conditional_losses_174795features/subject_idfeatures/supplier_id"?
???
FullArgSpec+
args#? 
jself

jfeatures

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
P__inference_supplier_recommender_layer_call_and_return_conditional_losses_174517
subject_idsupplier_id"?
???
FullArgSpec+
args#? 
jself

jfeatures

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
P__inference_supplier_recommender_layer_call_and_return_conditional_losses_174555
subject_idsupplier_id"?
???
FullArgSpec+
args#? 
jself

jfeatures

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
v
0
1
2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
?
Etrace_0
Ftrace_1
Gtrace_2
Htrace_32?
.__inference_ranking_model_layer_call_fn_174009
.__inference_ranking_model_layer_call_fn_174833
.__inference_ranking_model_layer_call_fn_174871
.__inference_ranking_model_layer_call_fn_174165?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zEtrace_0zFtrace_1zGtrace_2zHtrace_3
?
Itrace_0
Jtrace_1
Ktrace_2
Ltrace_32?
I__inference_ranking_model_layer_call_and_return_conditional_losses_174930
I__inference_ranking_model_layer_call_and_return_conditional_losses_174989
I__inference_ranking_model_layer_call_and_return_conditional_losses_174207
I__inference_ranking_model_layer_call_and_return_conditional_losses_174249?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zItrace_0zJtrace_1zKtrace_2zLtrace_3
?
Mlayer-0
Nlayer_with_weights-0
Nlayer-1
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
Ulayer-0
Vlayer_with_weights-0
Vlayer-1
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
]layer_with_weights-0
]layer-0
^layer_with_weights-1
^layer-1
_layer_with_weights-2
_layer-2
`layer_with_weights-3
`layer-3
alayer_with_weights-4
alayer-4
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses"
_tf_keras_sequential
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec\
argsT?Q
jself
jlabels
jpredictions
jsample_weight

jtraining
jcompute_metrics
varargs
 
varkw
 
defaults?

 
p 
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec\
argsT?Q
jself
jlabels
jpredictions
jsample_weight

jtraining
jcompute_metrics
varargs
 
varkw
 
defaults?

 
p 
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
'
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?B?
$__inference_signature_wrapper_174601
subject_idsupplier_id"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
N
m	variables
n	keras_api
	ototal
	pcount"
_tf_keras_metric
 "
trackable_list_wrapper
5
,0
-1
.2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
.__inference_ranking_model_layer_call_fn_174009input_1input_2"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
.__inference_ranking_model_layer_call_fn_174833inputs/0inputs/1"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
.__inference_ranking_model_layer_call_fn_174871inputs/0inputs/1"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
.__inference_ranking_model_layer_call_fn_174165input_1input_2"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
I__inference_ranking_model_layer_call_and_return_conditional_losses_174930inputs/0inputs/1"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
I__inference_ranking_model_layer_call_and_return_conditional_losses_174989inputs/0inputs/1"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
I__inference_ranking_model_layer_call_and_return_conditional_losses_174207input_1input_2"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
I__inference_ranking_model_layer_call_and_return_conditional_losses_174249input_1input_2"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:
q	keras_api
rlookup_table"
_tf_keras_layer
?
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
?
~trace_0
trace_1
?trace_2
?trace_32?
+__inference_sequential_layer_call_fn_173418
+__inference_sequential_layer_call_fn_175000
+__inference_sequential_layer_call_fn_175011
+__inference_sequential_layer_call_fn_173470?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z~trace_0ztrace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
F__inference_sequential_layer_call_and_return_conditional_losses_175024
F__inference_sequential_layer_call_and_return_conditional_losses_175037
F__inference_sequential_layer_call_and_return_conditional_losses_173481
F__inference_sequential_layer_call_and_return_conditional_losses_173492?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
<
?	keras_api
?lookup_table"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
-__inference_sequential_1_layer_call_fn_173526
-__inference_sequential_1_layer_call_fn_175048
-__inference_sequential_1_layer_call_fn_175059
-__inference_sequential_1_layer_call_fn_173578?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
H__inference_sequential_1_layer_call_and_return_conditional_losses_175072
H__inference_sequential_1_layer_call_and_return_conditional_losses_175085
H__inference_sequential_1_layer_call_and_return_conditional_losses_173589
H__inference_sequential_1_layer_call_and_return_conditional_losses_173600?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
-__inference_sequential_2_layer_call_fn_173715
-__inference_sequential_2_layer_call_fn_175110
-__inference_sequential_2_layer_call_fn_175135
-__inference_sequential_2_layer_call_fn_173869?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
H__inference_sequential_2_layer_call_and_return_conditional_losses_175173
H__inference_sequential_2_layer_call_and_return_conditional_losses_175211
H__inference_sequential_2_layer_call_and_return_conditional_losses_173898
H__inference_sequential_2_layer_call_and_return_conditional_losses_173927?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
=
?root_mean_squared_error"
trackable_dict_wrapper
.
o0
p1"
trackable_list_wrapper
-
m	variables"
_generic_user_object
:  (2total
:  (2count
"
_generic_user_object
j
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jtf.StaticHashTable
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_embedding_layer_call_fn_175218?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
E__inference_embedding_layer_call_and_return_conditional_losses_175227?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
+__inference_sequential_layer_call_fn_173418string_lookup_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
+__inference_sequential_layer_call_fn_175000inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
+__inference_sequential_layer_call_fn_175011inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
+__inference_sequential_layer_call_fn_173470string_lookup_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
F__inference_sequential_layer_call_and_return_conditional_losses_175024inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
F__inference_sequential_layer_call_and_return_conditional_losses_175037inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
F__inference_sequential_layer_call_and_return_conditional_losses_173481string_lookup_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
F__inference_sequential_layer_call_and_return_conditional_losses_173492string_lookup_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
"
_generic_user_object
j
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jtf.StaticHashTable
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
,__inference_embedding_1_layer_call_fn_175234?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
G__inference_embedding_1_layer_call_and_return_conditional_losses_175243?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
-__inference_sequential_1_layer_call_fn_173526string_lookup_1_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
-__inference_sequential_1_layer_call_fn_175048inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
-__inference_sequential_1_layer_call_fn_175059inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
-__inference_sequential_1_layer_call_fn_173578string_lookup_1_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
H__inference_sequential_1_layer_call_and_return_conditional_losses_175072inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
H__inference_sequential_1_layer_call_and_return_conditional_losses_175085inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
H__inference_sequential_1_layer_call_and_return_conditional_losses_173589string_lookup_1_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
H__inference_sequential_1_layer_call_and_return_conditional_losses_173600string_lookup_1_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
&__inference_dense_layer_call_fn_175252?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
A__inference_dense_layer_call_and_return_conditional_losses_175263?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_dense_1_layer_call_fn_175272?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_dense_1_layer_call_and_return_conditional_losses_175283?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_dense_2_layer_call_fn_175292?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_dense_2_layer_call_and_return_conditional_losses_175303?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_dense_3_layer_call_fn_175312?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_dense_3_layer_call_and_return_conditional_losses_175323?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_dense_4_layer_call_fn_175332?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_dense_4_layer_call_and_return_conditional_losses_175342?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
C
]0
^1
_2
`3
a4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
-__inference_sequential_2_layer_call_fn_173715dense_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
-__inference_sequential_2_layer_call_fn_175110inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
-__inference_sequential_2_layer_call_fn_175135inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
-__inference_sequential_2_layer_call_fn_173869dense_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
H__inference_sequential_2_layer_call_and_return_conditional_losses_175173inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
H__inference_sequential_2_layer_call_and_return_conditional_losses_175211inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
H__inference_sequential_2_layer_call_and_return_conditional_losses_173898dense_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
H__inference_sequential_2_layer_call_and_return_conditional_losses_173927dense_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
"
_generic_user_object
?
?trace_02?
__inference__creator_175347?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_175355?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_175360?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_embedding_layer_call_fn_175218inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_embedding_layer_call_and_return_conditional_losses_175227inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
"
_generic_user_object
?
?trace_02?
__inference__creator_175365?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_175373?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_175378?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
,__inference_embedding_1_layer_call_fn_175234inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
G__inference_embedding_1_layer_call_and_return_conditional_losses_175243inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
&__inference_dense_layer_call_fn_175252inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
A__inference_dense_layer_call_and_return_conditional_losses_175263inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_dense_1_layer_call_fn_175272inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_dense_1_layer_call_and_return_conditional_losses_175283inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_dense_2_layer_call_fn_175292inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_dense_2_layer_call_and_return_conditional_losses_175303inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_dense_3_layer_call_fn_175312inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_dense_3_layer_call_and_return_conditional_losses_175323inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_dense_4_layer_call_fn_175332inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_dense_4_layer_call_and_return_conditional_losses_175342inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference__creator_175347"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_175355"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_175360"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__creator_175365"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_175373"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_175378"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
,:*	?@2Adam/embedding/embeddings/m
-:+8@2Adam/embedding_1/embeddings/m
%:#
??2Adam/dense/kernel/m
:?2Adam/dense/bias/m
&:$	?@2Adam/dense_1/kernel/m
:@2Adam/dense_1/bias/m
%:#@ 2Adam/dense_2/kernel/m
: 2Adam/dense_2/bias/m
%:# 2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
%:#2Adam/dense_4/kernel/m
:2Adam/dense_4/bias/m
,:*	?@2Adam/embedding/embeddings/v
-:+8@2Adam/embedding_1/embeddings/v
%:#
??2Adam/dense/kernel/v
:?2Adam/dense/bias/v
&:$	?@2Adam/dense_1/kernel/v
:@2Adam/dense_1/bias/v
%:#@ 2Adam/dense_2/kernel/v
: 2Adam/dense_2/bias/v
%:# 2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v
%:#2Adam/dense_4/kernel/v
:2Adam/dense_4/bias/v
J
Constjtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant7
__inference__creator_175347?

? 
? "? 7
__inference__creator_175365?

? 
? "? 9
__inference__destroyer_175360?

? 
? "? 9
__inference__destroyer_175378?

? 
? "? B
__inference__initializer_175355r???

? 
? "? C
__inference__initializer_175373 ????

? 
? "? ?
!__inference__wrapped_model_173384?r???t?q
j?g
e?b
.

subject_id ?

subject_id?????????
0
supplier_id!?
supplier_id?????????
? "3?0
.
output_1"?
output_1??????????
C__inference_dense_1_layer_call_and_return_conditional_losses_175283]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? |
(__inference_dense_1_layer_call_fn_175272P0?-
&?#
!?
inputs??????????
? "??????????@?
C__inference_dense_2_layer_call_and_return_conditional_losses_175303\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0????????? 
? {
(__inference_dense_2_layer_call_fn_175292O/?,
%?"
 ?
inputs?????????@
? "?????????? ?
C__inference_dense_3_layer_call_and_return_conditional_losses_175323\/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? {
(__inference_dense_3_layer_call_fn_175312O/?,
%?"
 ?
inputs????????? 
? "???????????
C__inference_dense_4_layer_call_and_return_conditional_losses_175342\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_dense_4_layer_call_fn_175332O/?,
%?"
 ?
inputs?????????
? "???????????
A__inference_dense_layer_call_and_return_conditional_losses_175263^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
&__inference_dense_layer_call_fn_175252Q0?-
&?#
!?
inputs??????????
? "????????????
G__inference_embedding_1_layer_call_and_return_conditional_losses_175243W+?(
!?
?
inputs?????????	
? "%?"
?
0?????????@
? z
,__inference_embedding_1_layer_call_fn_175234J+?(
!?
?
inputs?????????	
? "??????????@?
E__inference_embedding_layer_call_and_return_conditional_losses_175227W+?(
!?
?
inputs?????????	
? "%?"
?
0?????????@
? x
*__inference_embedding_layer_call_fn_175218J+?(
!?
?
inputs?????????	
? "??????????@?
I__inference_ranking_model_layer_call_and_return_conditional_losses_174207?r???T?Q
J?G
A?>
?
input_1?????????
?
input_2?????????
p 
? "%?"
?
0?????????
? ?
I__inference_ranking_model_layer_call_and_return_conditional_losses_174249?r???T?Q
J?G
A?>
?
input_1?????????
?
input_2?????????
p
? "%?"
?
0?????????
? ?
I__inference_ranking_model_layer_call_and_return_conditional_losses_174930?r???V?S
L?I
C?@
?
inputs/0?????????
?
inputs/1?????????
p 
? "%?"
?
0?????????
? ?
I__inference_ranking_model_layer_call_and_return_conditional_losses_174989?r???V?S
L?I
C?@
?
inputs/0?????????
?
inputs/1?????????
p
? "%?"
?
0?????????
? ?
.__inference_ranking_model_layer_call_fn_174009?r???T?Q
J?G
A?>
?
input_1?????????
?
input_2?????????
p 
? "???????????
.__inference_ranking_model_layer_call_fn_174165?r???T?Q
J?G
A?>
?
input_1?????????
?
input_2?????????
p
? "???????????
.__inference_ranking_model_layer_call_fn_174833?r???V?S
L?I
C?@
?
inputs/0?????????
?
inputs/1?????????
p 
? "???????????
.__inference_ranking_model_layer_call_fn_174871?r???V?S
L?I
C?@
?
inputs/0?????????
?
inputs/1?????????
p
? "???????????
H__inference_sequential_1_layer_call_and_return_conditional_losses_173589r??B??
8?5
+?(
string_lookup_1_input?????????
p 

 
? "%?"
?
0?????????@
? ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_173600r??B??
8?5
+?(
string_lookup_1_input?????????
p

 
? "%?"
?
0?????????@
? ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_175072c??3?0
)?&
?
inputs?????????
p 

 
? "%?"
?
0?????????@
? ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_175085c??3?0
)?&
?
inputs?????????
p

 
? "%?"
?
0?????????@
? ?
-__inference_sequential_1_layer_call_fn_173526e??B??
8?5
+?(
string_lookup_1_input?????????
p 

 
? "??????????@?
-__inference_sequential_1_layer_call_fn_173578e??B??
8?5
+?(
string_lookup_1_input?????????
p

 
? "??????????@?
-__inference_sequential_1_layer_call_fn_175048V??3?0
)?&
?
inputs?????????
p 

 
? "??????????@?
-__inference_sequential_1_layer_call_fn_175059V??3?0
)?&
?
inputs?????????
p

 
? "??????????@?
H__inference_sequential_2_layer_call_and_return_conditional_losses_173898r
=?:
3?0
&?#
dense_input??????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_2_layer_call_and_return_conditional_losses_173927r
=?:
3?0
&?#
dense_input??????????
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_2_layer_call_and_return_conditional_losses_175173m
8?5
.?+
!?
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_2_layer_call_and_return_conditional_losses_175211m
8?5
.?+
!?
inputs??????????
p

 
? "%?"
?
0?????????
? ?
-__inference_sequential_2_layer_call_fn_173715e
=?:
3?0
&?#
dense_input??????????
p 

 
? "???????????
-__inference_sequential_2_layer_call_fn_173869e
=?:
3?0
&?#
dense_input??????????
p

 
? "???????????
-__inference_sequential_2_layer_call_fn_175110`
8?5
.?+
!?
inputs??????????
p 

 
? "???????????
-__inference_sequential_2_layer_call_fn_175135`
8?5
.?+
!?
inputs??????????
p

 
? "???????????
F__inference_sequential_layer_call_and_return_conditional_losses_173481or?@?=
6?3
)?&
string_lookup_input?????????
p 

 
? "%?"
?
0?????????@
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_173492or?@?=
6?3
)?&
string_lookup_input?????????
p

 
? "%?"
?
0?????????@
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_175024br?3?0
)?&
?
inputs?????????
p 

 
? "%?"
?
0?????????@
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_175037br?3?0
)?&
?
inputs?????????
p

 
? "%?"
?
0?????????@
? ?
+__inference_sequential_layer_call_fn_173418br?@?=
6?3
)?&
string_lookup_input?????????
p 

 
? "??????????@?
+__inference_sequential_layer_call_fn_173470br?@?=
6?3
)?&
string_lookup_input?????????
p

 
? "??????????@?
+__inference_sequential_layer_call_fn_175000Ur?3?0
)?&
?
inputs?????????
p 

 
? "??????????@?
+__inference_sequential_layer_call_fn_175011Ur?3?0
)?&
?
inputs?????????
p

 
? "??????????@?
$__inference_signature_wrapper_174601?r???o?l
? 
e?b
.

subject_id ?

subject_id?????????
0
supplier_id!?
supplier_id?????????"3?0
.
output_1"?
output_1??????????
P__inference_supplier_recommender_layer_call_and_return_conditional_losses_174517?r???x?u
n?k
e?b
.

subject_id ?

subject_id?????????
0
supplier_id!?
supplier_id?????????
p 
? "%?"
?
0?????????
? ?
P__inference_supplier_recommender_layer_call_and_return_conditional_losses_174555?r???x?u
n?k
e?b
.

subject_id ?

subject_id?????????
0
supplier_id!?
supplier_id?????????
p
? "%?"
?
0?????????
? ?
P__inference_supplier_recommender_layer_call_and_return_conditional_losses_174736?r??????
??}
w?t
7

subject_id)?&
features/subject_id?????????
9
supplier_id*?'
features/supplier_id?????????
p 
? "%?"
?
0?????????
? ?
P__inference_supplier_recommender_layer_call_and_return_conditional_losses_174795?r??????
??}
w?t
7

subject_id)?&
features/subject_id?????????
9
supplier_id*?'
features/supplier_id?????????
p
? "%?"
?
0?????????
? ?
5__inference_supplier_recommender_layer_call_fn_174327?r???x?u
n?k
e?b
.

subject_id ?

subject_id?????????
0
supplier_id!?
supplier_id?????????
p 
? "???????????
5__inference_supplier_recommender_layer_call_fn_174479?r???x?u
n?k
e?b
.

subject_id ?

subject_id?????????
0
supplier_id!?
supplier_id?????????
p
? "???????????
5__inference_supplier_recommender_layer_call_fn_174639?r??????
??}
w?t
7

subject_id)?&
features/subject_id?????????
9
supplier_id*?'
features/supplier_id?????????
p 
? "???????????
5__inference_supplier_recommender_layer_call_fn_174677?r??????
??}
w?t
7

subject_id)?&
features/subject_id?????????
9
supplier_id*?'
features/supplier_id?????????
p
? "??????????