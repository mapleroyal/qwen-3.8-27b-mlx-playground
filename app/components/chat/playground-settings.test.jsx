import * as React from "react";
import { describe, expect, it, vi } from "vitest";

import {
  PlaygroundSettings,
  THINKING_LEVELS,
} from "@/components/chat/playground-settings";

function findElement(node, predicate) {
  if (!React.isValidElement(node)) {
    return null;
  }

  if (predicate(node)) {
    return node;
  }

  for (const child of React.Children.toArray(node.props.children)) {
    const match = findElement(child, predicate);
    if (match) {
      return match;
    }
  }

  return null;
}

function settingsTree({ contextLength, onSettingChange }) {
  return PlaygroundSettings({
    backendSwitchDisabled: false,
    conversationHasImage: false,
    idPrefix: "test",
    onBackendChange: vi.fn(),
    onReset: vi.fn(),
    onSettingChange,
    onSystemPromptChange: vi.fn(),
    runtime: {
      activeBackend: "dspark",
      backends: [
        { id: "dspark", label: "DSpark", description: "Text" },
        { id: "mtp", label: "MTP", description: "Vision" },
      ],
      contextLength,
      maxOutputTokens: contextLength,
      requestedBackend: "dspark",
      supports: { prefixCache: true, vision: false },
    },
    settings: {
      maxTokens: 262144,
      presencePenalty: 0,
      reasoningEffort: "medium",
      temperature: 1,
      topK: 20,
      topP: 0.95,
    },
    systemPrompt: "",
  });
}

describe("PlaygroundSettings", () => {
  it("maps Qwen's four thinking-effort stops directly", () => {
    const onSettingChange = vi.fn();
    const tree = settingsTree({ contextLength: 262144, onSettingChange });
    const control = findElement(
      tree,
      (element) => element.props.id === "test-thinking-effort",
    );

    expect(THINKING_LEVELS.map((level) => level.value)).toEqual([
      "off",
      "low",
      "medium",
      "xhigh",
    ]);
    expect(control?.props.value).toBe("medium");

    control.props.onChange("low");

    expect(onSettingChange).toHaveBeenCalledWith("reasoningEffort", "low");
  });

  it("exposes and directly populates the full context limit", () => {
    const contextLength = 262144;
    const onSettingChange = vi.fn();
    const tree = settingsTree({ contextLength, onSettingChange });
    const input = findElement(
      tree,
      (element) => element.props.id === "test-max-tokens",
    );
    const label = findElement(
      tree,
      (element) => element.props.htmlFor === "test-max-tokens",
    );
    const button = findElement(
      tree,
      (element) => element.props["aria-label"] === "Set Max output to 262,144",
    );

    expect(label?.props.children).toBe("Max output");
    expect(input?.props.max).toBe(262144);
    expect(button?.props.children).toBe("Max");
    expect(button?.props.title).toBe(
      `Use the full ${new Intl.NumberFormat().format(contextLength)}-token context limit`,
    );

    button.props.onClick();

    expect(onSettingChange).toHaveBeenCalledOnce();
    expect(onSettingChange).toHaveBeenCalledWith("maxTokens", 262144);
  });
});
