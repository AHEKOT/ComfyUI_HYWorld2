import { app } from "/scripts/app.js";

function firstGgufModel(node) {
    const widget = node.widgets?.find((item) => item.name === "llm_model");
    const values = widget?.options?.values;
    if (Array.isArray(values)) {
        return values.find((value) => typeof value === "string" && value.toLowerCase().endsWith(".gguf"))
            ?? values[0];
    }
    return widget?.value;
}

function wrapConfigure(nodeType, migrate) {
    const original = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
        if (Array.isArray(info?.widgets_values)) {
            const migrated = migrate.call(this, info.widgets_values);
            if (migrated) {
                info = { ...info, widgets_values: migrated };
                console.info(`[HYWorld2] Migrated legacy LLM widgets for ${info.type ?? this.type} to GGUF.`);
            }
        }
        return original?.apply(this, [info]);
    };
}

app.registerExtension({
    name: "hyworld2.gguf-widget-migration",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "HYWorld2Trajectories") {
            wrapConfigure(nodeType, function (values) {
                // Legacy order: ... model, quantization, max_edge, anchor, topk.
                if (typeof values[7] !== "string") {
                    return null;
                }
                return [
                    values[0], values[1], values[2], values[3], values[4], values[5],
                    firstGgufModel(this), values[8], 8192, -1, values[9], values[10],
                ];
            });
        }

        if (nodeData.name === "HYWorld2WorldExpansion") {
            wrapConfigure(nodeType, function (values) {
                // Legacy order began with model, quantization, attention.
                if (typeof values[1] !== "string") {
                    return null;
                }
                return [
                    firstGgufModel(this), values[3], values[4], values[5], values[6],
                    8192, -1, values[7], values[8], values[9],
                    values[10] ?? "", values[11] ?? 0, values[12] ?? 0,
                ];
            });
        }
    },
});
