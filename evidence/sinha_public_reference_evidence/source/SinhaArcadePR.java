import com.arcadedb.database.*;
import com.arcadedb.graph.*;
import com.arcadedb.graph.olap.*;
import com.arcadedb.schema.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;

/** Benchmark wrapper around the unmodified released ArcadeDB PageRank kernel. */
public class SinhaArcadePR {
  public static void main(String[] args) throws Exception {
    Path data = Path.of(args[0]), destination = Path.of(args[1]).toAbsolutePath().normalize();
    if (!destination.startsWith(Path.of("/kaggle/working")) || Files.exists(destination))
      throw new IllegalArgumentException("A fresh database inside the benchmark workspace is required");
    long started = System.nanoTime();
    Database db = new DatabaseFactory(destination.toString()).create();
    db.begin();
    db.getSchema().createVertexType("Vertex", 8);
    db.getSchema().createEdgeType("EDGE", 8);
    db.getSchema().getType("Vertex").createProperty("VID", Type.LONG);
    db.getSchema().getType("Vertex").createTypeIndex(Schema.INDEX_TYPE.HASH, true, "VID");
    db.getSchema().getType("EDGE").createProperty("WEIGHT", Type.DOUBLE);
    db.commit();
    Map<Long, RID> ids = new HashMap<>(700000);
    db.begin();
    try (BufferedReader reader = Files.newBufferedReader(data.resolve("datagen-7_5-fb.v"))) {
      String line;
      while ((line = reader.readLine()) != null) {
        long id = Long.parseLong(line.trim());
        MutableVertex vertex = db.newVertex("Vertex");
        vertex.set("VID", id); vertex.save(); ids.put(id, vertex.getIdentity());
        if (ids.size() % 10000 == 0) { db.commit(); db.begin(); }
      }
    }
    db.commit();
    System.out.println("ARCADE VERTICES " + ids.size());
    GraphBatch importer = db.batch().withBatchSize(100000).withLightEdges(false).withWAL(false).build();
    long edges = 0;
    try (BufferedReader reader = new BufferedReader(new FileReader(data.resolve("datagen-7_5-fb.e").toFile()), 1<<20)) {
      String line;
      while ((line = reader.readLine()) != null) {
        String[] parts = line.split(" ");
        long from = Long.parseLong(parts[0]), to = Long.parseLong(parts[1]);
        if (!ids.containsKey(from) || !ids.containsKey(to)) throw new IllegalArgumentException("Unknown vertex");
        importer.newEdge(ids.get(from), "EDGE", ids.get(to), "WEIGHT", Double.parseDouble(parts[2]));
        if (++edges % 5000000 == 0) System.out.println("ARCADE EDGES " + edges);
      }
    }
    importer.close();
    GraphAnalyticalView view = GraphAnalyticalView.builder(db).withName("benchmark")
        .withVertexTypes("Vertex").withEdgeTypes("EDGE").withEdgeProperties("WEIGHT").build();
    if (!GraphTraversalProviderRegistry.awaitAll(db, 120, java.util.concurrent.TimeUnit.SECONDS))
      throw new IllegalStateException("Graph view is not ready");
    double preparation = (System.nanoTime()-started)/1e9;
    double[] scores = GraphAlgorithms.pageRank(view, 0.85, 10, Vertex.DIRECTION.BOTH, "EDGE");
    double[] samples = new double[31];
    for (int i=0; i<samples.length; i++) {
      long before = System.nanoTime();
      scores = GraphAlgorithms.pageRank(view, 0.85, 10, Vertex.DIRECTION.BOTH, "EDGE");
      samples[i] = (System.nanoTime()-before)/1e6;
    }
    db.begin();
    try (DataOutputStream stream = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(args[2])))) {
      for (int i=0; i<scores.length; i++) {
        long id = view.getNodeMapping().getRID(i).asVertex().getLong("VID");
        stream.writeLong(id); stream.writeDouble(scores[i]);
      }
    } finally { db.rollback(); }
    String json = "{\"vertices\":"+scores.length+",\"edges\":"+edges+",\"prepare_seconds\":"+preparation
        +",\"cpu_processors\":"+Runtime.getRuntime().availableProcessors()+",\"samples_ms\":"+Arrays.toString(samples)+"}";
    Files.writeString(Path.of(args[3]), json);
    db.close();
    System.out.println("ARCADE COMPLETE " + json);
  }
}
