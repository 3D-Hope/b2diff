import { useState, useEffect, useCallback, useRef } from "react";
import { FixedSizeList as List } from "react-window";

const IMG_H = 220;
const CAPTION_H = 26;
const ROW_H = IMG_H + CAPTION_H + 12;
const SIDEBAR_W = 240;
const IDX_COL_W = 44;

export default function App() {
  const [runs, setRuns] = useState([]);
  const [selected, setSelected] = useState([]);
  const [imagesMap, setImagesMap] = useState({});
  const [search, setSearch] = useState("");
  const [imageSize, setImageSize] = useState(IMG_H);
  const [mainW, setMainW] = useState(window.innerWidth - SIDEBAR_W);
  const mainRef = useRef(null);
  const listRef = useRef(null);

  // Fetch run list and auto-select all
  useEffect(() => {
    fetch("/api/runs")
      .then((r) => r.json())
      .then((fetched) => {
        setRuns(fetched);
        setSelected(fetched);
        fetched.forEach((run) => {
          fetch(`/api/images/${run}`)
            .then((r) => r.json())
            .then((imgs) =>
              setImagesMap((prev) => ({ ...prev, [run]: imgs }))
            );
        });
      });
  }, []);

  // Observe main panel width
  useEffect(() => {
    const el = mainRef.current;
    if (!el) return;
    const obs = new ResizeObserver(([e]) => setMainW(e.contentRect.width));
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  // Toggle run selection and lazy-fetch its image list
  const toggleRun = useCallback(
    async (run) => {
      setSelected((prev) => {
        if (prev.includes(run)) return prev.filter((r) => r !== run);
        return [...prev, run];
      });
      if (!imagesMap[run]) {
        const imgs = await fetch(`/api/images/${run}`).then((r) => r.json());
        setImagesMap((prev) => ({ ...prev, [run]: imgs }));
      }
    },
    [imagesMap]
  );

  const clearAll = () => setSelected([]);
  const selectAll = () => {
    setSelected(runs);
    runs.forEach((run) => {
      if (!imagesMap[run]) {
        fetch(`/api/images/${run}`)
          .then((r) => r.json())
          .then((imgs) =>
            setImagesMap((prev) => ({ ...prev, [run]: imgs }))
          );
      }
    });
  };

  // Row count = max images across selected runs
  const rowCount =
    selected.length === 0
      ? 0
      : Math.max(...selected.map((r) => (imagesMap[r] || []).length));

  // Cell width per run
  const cellW =
    selected.length > 0
      ? Math.max(120, (mainW - IDX_COL_W - 8) / selected.length)
      : 0;

  const filteredRuns = runs.filter((r) =>
    r.toLowerCase().includes(search.toLowerCase())
  );

  const rowH = imageSize + CAPTION_H + 12;

  // Row renderer
  const Row = ({ index, style }) => {
    return (
      <div
        style={{
          ...style,
          display: "flex",
          alignItems: "flex-start",
          gap: 0,
          borderBottom: "1px solid #1e1e30",
        }}
      >
        {/* Row index */}
        <div
          style={{
            width: IDX_COL_W,
            flexShrink: 0,
            height: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#555",
            fontSize: 11,
            userSelect: "none",
          }}
        >
          {String(index).padStart(4, "0")}
        </div>

        {/* One cell per selected run */}
        {selected.map((run) => {
          const imgs = imagesMap[run] || [];
          const filename = imgs[index];
          return (
            <div
              key={run}
              style={{
                width: cellW,
                flexShrink: 0,
                padding: "4px 4px 0 4px",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
              }}
            >
              {filename ? (
                <>
                  <img
                    src={`/api/image/${encodeURIComponent(run)}/${encodeURIComponent(filename)}`}
                    loading="lazy"
                    style={{
                      height: imageSize,
                      width: "100%",
                      objectFit: "contain",
                      background: "#12122a",
                      cursor: "zoom-in",
                      borderRadius: 3,
                    }}
                    onClick={() =>
                      window.open(
                        `/api/image/${encodeURIComponent(run)}/${encodeURIComponent(filename)}`,
                        "_blank"
                      )
                    }
                    title={filename}
                    onError={(e) => {
                      e.target.style.background = "#2a1a1a";
                      e.target.alt = "error";
                    }}
                  />
                  <div
                    style={{
                      fontSize: 10,
                      color: "#688",
                      marginTop: 3,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                      width: "100%",
                      textAlign: "center",
                    }}
                  >
                    {filename.replace(/\.[^.]+$/, "")}
                  </div>
                </>
              ) : (
                <div
                  style={{
                    height: imageSize,
                    width: "100%",
                    background: "#1a1a2e",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    color: "#333",
                    fontSize: 12,
                    borderRadius: 3,
                  }}
                >
                  {imagesMap[run] ? "—" : "loading…"}
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden" }}>
      {/* ── Sidebar ── */}
      <aside
        style={{
          width: SIDEBAR_W,
          minWidth: SIDEBAR_W,
          background: "#12122a",
          borderRight: "1px solid #1e1e40",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}
      >
        <div style={{ padding: "12px 10px 8px", borderBottom: "1px solid #1e1e40" }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: "#a0a0e0", marginBottom: 8 }}>
            Runs ({runs.length})
          </div>
          <input
            placeholder="Search runs…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{
              width: "100%",
              padding: "5px 8px",
              background: "#1a1a36",
              border: "1px solid #2a2a50",
              borderRadius: 4,
              color: "#ddd",
              fontSize: 12,
              outline: "none",
            }}
          />
          <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
            <button onClick={selectAll} style={btnStyle("#1d3a1d", "#4caf50")}>
              All
            </button>
            <button onClick={clearAll} style={btnStyle("#3a1d1d", "#f44336")}>
              Clear
            </button>
          </div>
          {selected.length > 0 && (
            <div style={{ fontSize: 11, color: "#778", marginTop: 6 }}>
              {selected.length} selected · {rowCount} images each
            </div>
          )}
        </div>

        {/* Run list */}
        <div style={{ overflowY: "auto", flex: 1, padding: "6px 0" }}>
          {filteredRuns.map((run) => {
            const isSelected = selected.includes(run);
            return (
              <label
                key={run}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  padding: "5px 12px",
                  cursor: "pointer",
                  background: isSelected ? "#1a2a3a" : "transparent",
                  borderLeft: isSelected ? "3px solid #4fc3f7" : "3px solid transparent",
                  transition: "background 0.1s",
                }}
              >
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={() => toggleRun(run)}
                  style={{ accentColor: "#4fc3f7", cursor: "pointer" }}
                />
                <span
                  style={{
                    fontSize: 11,
                    color: isSelected ? "#cce" : "#778",
                    wordBreak: "break-all",
                    lineHeight: 1.4,
                  }}
                >
                  {run}
                </span>
              </label>
            );
          })}
        </div>

        {/* Image size slider */}
        <div
          style={{
            padding: "10px 12px",
            borderTop: "1px solid #1e1e40",
            fontSize: 11,
            color: "#778",
          }}
        >
          <div style={{ marginBottom: 4 }}>Image height: {imageSize}px</div>
          <input
            type="range"
            min={80}
            max={500}
            value={imageSize}
            onChange={(e) => {
              setImageSize(Number(e.target.value));
              listRef.current?.resetAfterIndex?.(0);
            }}
            style={{ width: "100%", accentColor: "#4fc3f7" }}
          />
        </div>
      </aside>

      {/* ── Main panel ── */}
      <div
        ref={mainRef}
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
          background: "#0f0f1a",
        }}
      >
        {selected.length === 0 ? (
          <div
            style={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#334",
              fontSize: 16,
            }}
          >
            ← Select runs to compare
          </div>
        ) : (
          <>
            {/* Sticky header */}
            <div
              style={{
                display: "flex",
                background: "#12122a",
                borderBottom: "2px solid #1e1e40",
                flexShrink: 0,
              }}
            >
              <div style={{ width: IDX_COL_W, flexShrink: 0 }} />
              {selected.map((run) => (
                <div
                  key={run}
                  style={{
                    width: cellW,
                    flexShrink: 0,
                    padding: "8px 4px",
                    fontSize: 11,
                    fontWeight: 600,
                    color: "#4fc3f7",
                    textAlign: "center",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    borderLeft: "1px solid #1e1e40",
                  }}
                  title={run}
                >
                  {run}
                </div>
              ))}
            </div>

            {/* Virtual list */}
            <div style={{ flex: 1, overflow: "hidden" }}>
              <List
                ref={listRef}
                height={window.innerHeight - 80}
                itemCount={rowCount}
                itemSize={rowH}
                width={mainW}
                overscanCount={3}
              >
                {Row}
              </List>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function btnStyle(bg, color) {
  return {
    flex: 1,
    padding: "4px 0",
    background: bg,
    border: `1px solid ${color}44`,
    borderRadius: 4,
    color,
    fontSize: 11,
    cursor: "pointer",
    fontWeight: 600,
  };
}
