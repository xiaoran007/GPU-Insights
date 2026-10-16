import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import "./workbench/workbench.css";
import WorkbenchRoot from "./workbench/Workbench";

createRoot(document.getElementById("root")!).render(<StrictMode><WorkbenchRoot /></StrictMode>);
