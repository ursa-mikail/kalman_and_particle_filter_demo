package main

import (
	"database/sql"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"time"

	_ "github.com/lib/pq"
)

var db *sql.DB

func initDB() {
	url := os.Getenv("DATABASE_URL")
	if url == "" {
		return
	}
	for i := 0; i < 30; i++ {
		d, err := sql.Open("postgres", url)
		if err == nil && d.Ping() == nil {
			if _, err = d.Exec(`CREATE TABLE IF NOT EXISTS runs(
				id SERIAL PRIMARY KEY, created_at TIMESTAMPTZ DEFAULT now(),
				params JSONB, summary JSONB, result JSONB)`); err == nil {
				db = d
				log.Println("postgres ready")
				return
			}
		}
		time.Sleep(time.Second)
	}
	log.Println("postgres unavailable: running without history")
}

func js(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

func main() {
	initDB()
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/health", func(w http.ResponseWriter, r *http.Request) {
		js(w, map[string]any{"ok": true, "db": db != nil})
	})
	mux.HandleFunc("POST /api/run", func(w http.ResponseWriter, r *http.Request) {
		var p Params
		if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		res := Run(p)
		if db != nil {
			type row struct {
				ID       string  `json:"id"`
				RMSE     float64 `json:"rmse"`
				Ms       float64 `json:"ms"`
			}
			var sum []row
			for _, f := range res.Filters {
				sum = append(sum, row{f.ID, f.RMSEPos, f.Ms})
			}
			pb, _ := json.Marshal(res.Params)
			sb, _ := json.Marshal(sum)
			rb, _ := json.Marshal(res)
			if err := db.QueryRow(`INSERT INTO runs(params,summary,result) VALUES($1,$2,$3) RETURNING id`, pb, sb, rb).Scan(&res.RunID); err != nil {
				log.Println("save:", err)
			}
		}
		js(w, res)
	})
	mux.HandleFunc("GET /api/runs", func(w http.ResponseWriter, r *http.Request) {
		out := []map[string]any{}
		if db != nil {
			rows, err := db.Query(`SELECT id, created_at, params, summary FROM runs ORDER BY id DESC LIMIT 20`)
			if err == nil {
				defer rows.Close()
				for rows.Next() {
					var id int
					var t time.Time
					var p, s []byte
					rows.Scan(&id, &t, &p, &s)
					out = append(out, map[string]any{"id": id, "at": t, "params": json.RawMessage(p), "summary": json.RawMessage(s)})
				}
			}
		}
		js(w, out)
	})
	mux.HandleFunc("GET /api/runs/{id}", func(w http.ResponseWriter, r *http.Request) {
		if db == nil {
			http.NotFound(w, r)
			return
		}
		var b []byte
		if err := db.QueryRow(`SELECT result FROM runs WHERE id=$1`, r.PathValue("id")).Scan(&b); err != nil {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write(b)
	})
	log.Println("api on :8080")
	log.Fatal(http.ListenAndServe(":8080", mux))
}
