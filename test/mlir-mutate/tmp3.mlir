"builtin.module"() ({
  "func.func"() <{function_type = (i4, i4) -> i4, sym_name = "tmp"}> ({
  ^bb0(%arg0: i4, %arg1: i4):
    %0 = "hw.constant"() {value = -2 : i4} : () -> i4
    %1 = "comb.and"(%0, %arg0) <{twoState}> : (i4, i4) -> i4
    %2 = "comb.and"(%0, %arg1) <{twoState}> : (i4, i4) -> i4
    %3 = "comb.and"(%2, %1) <{twoState}> : (i4, i4) -> i4
    "func.return"(%3) : (i4) -> ()
  }) : () -> ()
}) : () -> ()
