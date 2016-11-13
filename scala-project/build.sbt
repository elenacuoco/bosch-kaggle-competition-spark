name := "NewXGBoost" 
version := "1.0" 
scalaVersion := "2.11.8" 
resolvers += Resolver.mavenLocal 
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.0.0" 
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.0.0" 
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.0.0" 
libraryDependencies += "org.apache.spark" %% "spark-yarn" % "2.0.0" 
libraryDependencies += "ml.dmlc" % "xgboost4j-spark" % "0.7" 
libraryDependencies += "ml.dmlc" % "xgboost4j"  % "0.7" 
ivyScala := ivyScala.value map { _.copy(overrideScalaVersion = true)}
