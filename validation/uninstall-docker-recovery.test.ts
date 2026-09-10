// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterAll, afterEach, describe, expect, it, vi } from "vitest";
import { withProvenManagedGatewayProcess } from "../../../../test/support/uninstall-managed-gateway-test-support";

import { type RunResult, runUninstallPlan } from "./run-plan";

const TEST_HOME = fs.mkdtempSync(path.join(os.tmpdir(), "nemoclaw-uninstall-docker-recovery-"));
const RETRY = "After 'docker info' succeeds, rerun 'nemoclaw uninstall' with the same options.";
const WSL_SETTINGS = "Docker Desktop > Settings > Resources > WSL integration";

function ok(stdout = ""): RunResult {
  return { status: 0, stdout, stderr: "" };
}

function uninstall(dockerInstalled: boolean, dockerStatus: number | null, removeStatus = 1) {
  const warnings: string[] = [];
  const logs: string[] = [];
  const rmSync = vi.fn();
  const responses: Record<string, RunResult> = {
    "openshell gateway list": ok(JSON.stringify([{ name: "nemoclaw" }])),
    "openshell gateway remove": { status: removeStatus, stdout: "", stderr: "" },
  };
  const run = vi.fn((command: string, args: string[]) =>
    responses[[command, ...args.slice(0, 2)].join(" ")] ?? ok(),
  );
  const runDocker = vi.fn((args: string[]) =>
    args[0] === "info"
      ? { status: dockerStatus, stdout: "ACCESS_TOKEN=must-not-be-logged", stderr: "secret-token" }
      : ok(),
  );
  const result = runUninstallPlan(
    { assumeYes: true, deleteModels: false, keepOpenShell: true },
    withProvenManagedGatewayProcess({
      isPortFree: () => true,
      resolveGatewayTeardownAuthority: ({ gatewayName, gatewayPort }) => ({
        gatewayName,
        gatewayPort,
        mode: "nemoclaw-managed",
        source: "packaged-service",
        endpoint: null,
        stateDir: null,
        supervisor: null,
        requiredCapabilities: [],
      }),
      commandExists: (command) => command !== "pgrep" && (command !== "docker" || dockerInstalled),
      env: { HOME: TEST_HOME, TMPDIR: os.tmpdir() },
      error: (line) => warnings.push(line),
      existsSync: () => false,
      isTty: false,
      log: (line) => logs.push(line),
      rmSync,
      run,
      runDocker,
    }),
  );
  return { result, output: warnings.join("\n"), logs, rmSync, run, runDocker };
}

function expectPreservedState(outcome: ReturnType<typeof uninstall>): void {
  expect(outcome.result.exitCode).toBe(1);
  expect(outcome.output).toContain("openshell gateway remove failed (exit 1).");
  expect(outcome.rmSync).not.toHaveBeenCalled();
  expect(outcome.logs).not.toContain("[3/6] NemoClaw CLI");
  expect(outcome.run.mock.calls.some(([command, args]) => command === "openshell" && args[1] === "destroy")).toBe(false);
  expect(outcome.output).not.toContain("must-not-be-logged");
  expect(outcome.output).not.toContain("secret-token");
}

afterAll(() => fs.rmSync(TEST_HOME, { recursive: true, force: true }));
afterEach(() => vi.unstubAllEnvs());

describe("uninstall Docker recovery (#11438)", () => {
  it("identifies a missing Docker command and preserves retry state", () => {
    const outcome = uninstall(false, 0);
    expectPreservedState(outcome);
    expect(outcome.output).toContain("docker command not found.");
    expect(outcome.output).toContain(WSL_SETTINGS);
    expect(outcome.output).toContain(RETRY);
    expect(outcome.runDocker).not.toHaveBeenCalled();
  });

  it.each([
    ["unreachable", 1],
    ["timed out", null],
  ] as const)("reports recovery when the Docker probe is %s", (_name, status) => {
    const outcome = uninstall(true, status);
    expectPreservedState(outcome);
    expect(outcome.output).toContain("Docker is unavailable.");
    expect(outcome.output).toContain(WSL_SETTINGS);
    expect(outcome.output).toContain(RETRY);
    expect(outcome.runDocker.mock.calls.filter(([args]) => args[0] === "info")).toHaveLength(1);
    expect(outcome.runDocker).toHaveBeenCalledWith(
      ["info"],
      expect.objectContaining({ timeout: 5_000, stdio: "ignore" }),
    );
  });

  it("retains the original failure when Docker is healthy", () => {
    const outcome = uninstall(true, 0);
    expectPreservedState(outcome);
    expect(outcome.output).not.toContain(RETRY);
    expect(outcome.output).not.toContain(WSL_SETTINGS);
    expect(outcome.runDocker.mock.calls.filter(([args]) => args[0] === "info")).toHaveLength(1);
    expect(outcome.runDocker).toHaveBeenCalledWith(
      ["info"],
      expect.objectContaining({ timeout: 5_000, stdio: "ignore" }),
    );
  });

  it("does not require Docker when registration removal succeeds", () => {
    const outcome = uninstall(false, 0, 0);
    expect(outcome.result.exitCode).toBe(0);
    expect(outcome.output).not.toContain(RETRY);
    expect(outcome.output).not.toContain(WSL_SETTINGS);
    expect(outcome.runDocker).not.toHaveBeenCalled();
  });

  it("does not add a recovery probe after successful registration removal", () => {
    const outcome = uninstall(true, 0, 0);
    expect(outcome.result.exitCode).toBe(0);
    expect(outcome.output).not.toContain(RETRY);
    expect(outcome.runDocker).not.toHaveBeenCalledWith(
      ["info"],
      expect.objectContaining({ timeout: 5_000 }),
    );
  });
});
